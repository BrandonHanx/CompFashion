import numpy as np
import PIL
import torch
import torch.nn.functional as F
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
        self.cat_type = cat_type
        self.data = []

        self.name = self.get_name()
        self.data = read_json(self.get_file_name())

        self.vocab_type = vocab
        if vocab not in ["init", "two-hot"]:
            vocab_file = f"{path}/{vocab}_vocab.npy"
            self.vocab = np_loader(vocab_file)

        split_file = f"{path}/split_crop_{split}.json"

        self.all_img_names = read_json(split_file)
        if sub_cats is not None:
            self.data = [x for x in self.data if x["target_cls"] in sub_cats]
        self.all_img_ids = {
            self.all_img_names[x]: x for x in range(len(self.all_img_names))
        }

    def get_name(self):
        return f"FashionPedia.{self.cat_type}.dict.{self.split}"

    def get_file_name(self):
        return f"{self.path}/{self.cat_type}_triplets_dict_{self.split}.json"

    def __getitem__(self, idx):
        source_img_name = self.data[idx]["candidate"]
        target_img_name = self.data[idx]["target"]
        meta_info = {
            "source_img_id": self.all_img_ids[source_img_name],
            "target_img_id": self.all_img_ids[target_img_name],
            "original_caption": " ".join(self.data[idx]["captions"]).lower(),
        }
        source_image = self.get_img(source_img_name)
        target_image = self.get_img(target_img_name)
        if self.vocab_type == "init":
            text = np.array(self.data[idx]["wv"])
        elif self.vocab_type == "two-hot":
            candidate_onehot = F.one_hot(
                torch.tensor(int(self.data[idx]["candidate_cls"])), num_classes=27
            )
            target_onehot = F.one_hot(
                torch.tensor(int(self.data[idx]["target_cls"])), num_classes=27
            )
            if self.cat_type == "outfit":
                text = torch.cat([candidate_onehot, target_onehot])
            elif self.cat_type == "outfit_wa":
                attribute_hot = torch.zeros(365)
                attribute = self.data[idx]["captions"]
                if len(attribute) > 0:
                    attribute_hot[attribute] = 1
                text = torch.cat([candidate_onehot, target_onehot, attribute_hot])
            else:
                NotImplementedError
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

    def get_specific_imgs(self, spec_list):
        imgs = []
        for x in spec_list:
            imgs.append(self.get_img(self.all_img_names[x]))
        return torch.stack(imgs)
