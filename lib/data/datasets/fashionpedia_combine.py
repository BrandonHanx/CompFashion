import numpy as np
import PIL
from torch.utils.data import Dataset

from lib.utils.directory import np_loader, read_json


class FashionPediaCombine(Dataset):
    """FashionPedia dataset."""

    def __init__(
        self,
        path,
        transform=None,
        vocab="clip",
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.transform = transform
        self.data = []
        self.name = "FashionPedia.combine.train"

        self.comp_data = read_json(f"{path}/comp_triplets_dict_train.json")
        self.outfit_data = read_json(f"{path}/outfit_triplets_dict_train.json")

        self.vocab = None
        if vocab != "init":
            vocab_file = f"{path}/{vocab}_vocab.npy"
            self.vocab = np_loader(vocab_file)

        all_img_names = read_json(f"{path}/split_crop_train.json")
        all_img_id_dict = {all_img_names[x]: x for x in range(len(all_img_names))}

        outfit_img_ids = []
        for triplet in self.outfit_data:
            outfit_img_ids.append(all_img_id_dict[triplet["candidate"]])
        outfit_img_ids = np.array(outfit_img_ids)

        quintuple_map = []
        for comp_idx, comp_triplet in enumerate(self.comp_data):
            candidate_img_id = all_img_id_dict[comp_triplet["candidate"]]
            outfit_idxs = np.where(outfit_img_ids == candidate_img_id)[0]
            if len(outfit_idxs) == 0:
                continue
            quintuple_map.append(dict(comp_idx=comp_idx, outfit_idxs=outfit_idxs))
        self.quintuple_map = quintuple_map

    def __getitem__(self, idx):
        quintuple = self.quintuple_map[idx]
        comp_triplet = self.comp_data[quintuple["comp_idx"]]
        outfit_idx = np.random.choice(quintuple["outfit_idxs"], 1)[0]
        outfit_triplet = self.outfit_data[outfit_idx]

        source_image = self.get_img(comp_triplet["candidate"])
        comp_target_image = self.get_img(comp_triplet["target"])
        outfit_target_image = self.get_img(outfit_triplet["target"])

        if self.vocab is None:
            comp_text = np.array(comp_triplet["wv"])
            outfit_text = np.array(outfit_triplet["wv"])
        else:
            comp_text = self.vocab[comp_triplet["wv"]]
            outfit_text = self.vocab[outfit_triplet["wv"]]

        meta_info = {
            "source_image": comp_triplet["candidate"],
            "comp_target_image": comp_triplet["target"],
            "outfit_target_image": outfit_triplet["target"],
            "comp_captions": comp_triplet["captions"],
            "outfit_captions": outfit_triplet["captions"],
        }

        return (
            source_image,
            comp_target_image,
            outfit_target_image,
            comp_text,
            outfit_text,
            meta_info,
        )

    def __len__(self):
        return len(self.quintuple_map)

    def get_img(self, img_name):
        img_path = f"{self.path}/crop_images/{img_name}"

        with open(img_path, "rb") as f:
            img = PIL.Image.open(f)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
