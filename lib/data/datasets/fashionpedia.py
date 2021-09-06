import numpy as np
import PIL
import torch
from torch.utils.data import Dataset

from lib.utils.directory import np_loader, read_json


class FashionPedia(Dataset):
    """FashionPedia dataset."""

    def __init__(
        self,
        path,
        split="train",
        cat_type="comp",
        transform=None,
        vocab="glove",
        sub_cats=None,
    ):
        super().__init__()
        self.path = path
        self.transform = transform
        self.split = split
        self.data = []
        self.name = f"FashionPedia.{cat_type}.dict.{split}"

        caps_file = f"{path}/{cat_type}_triplets_dict_{split}.json"
        self.data = read_json(caps_file)

        self.vocab = None
        if vocab != "init":
            vocab_file = f"{path}/{vocab}_vocab.npy"
            self.vocab = np_loader(vocab_file)

        split_file = f"{path}/split_crop_{split}.json"

        self.all_img_names = read_json(split_file)
        if sub_cats is not None:
            self.data = [x for x in self.data if x["target_cls"] in sub_cats]
        self.all_img_ids = {
            self.all_img_names[x]: x for x in range(len(self.all_img_names))
        }

    def __getitem__(self, idx):
        source_img_name = self.data[idx]["candidate"]
        target_img_name = self.data[idx]["target"]
        meta_info = {
            "source_img_id": self.all_img_ids[source_img_name],
            "target_img_id": self.all_img_ids[target_img_name],
            "original_caption": self.data[idx]["captions"],
        }
        source_image = self.get_img(source_img_name)
        target_image = self.get_img(target_img_name)
        if self.vocab is None:
            text = np.array(self.data[idx]["wv"])
        else:
            text = self.vocab[self.data[idx]["wv"]]

        return text, source_image, target_image, meta_info

    def __len__(self):
        return len(self.data)

    def get_img(self, img_name):
        img_path = f"{self.path}/crop_images/{img_name}"

        with open(img_path, "rb") as f:
            img = PIL.Image.open(f)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

    def get_all_imgs(self, batch_size):
        imgs = []
        for img_name in self.all_img_names:
            imgs.append(self.get_img(img_name))
            if len(imgs) == batch_size or img_name == self.all_img_names[-1]:
                batch_imgs = imgs
                imgs = []
                yield torch.stack(batch_imgs)
