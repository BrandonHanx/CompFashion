import numpy as np
import torch

from lib.utils.directory import read_json

from .fashionpedia import FashionPedia


class FashionPediaMultiTurn(FashionPedia):
    def __init__(
        self,
        path,
        split="train",
        cat_type="comp",
        transform=None,
        vocab="glove",
        turn=3,
    ):
        super().__init__(path, split, cat_type, transform, vocab)
        caps_file = f"{path}/{cat_type}_triplets_dict_{split}_turn{turn}.json"
        self.data = read_json(caps_file)

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
        if self.vocab_type == "init":
            text = [np.array(x) for x in self.data[idx]["wv"]]
        elif self.vocab_type == "two-hot":
            raise NotImplementedError
        else:
            text = [self.vocab[x] for x in self.data[idx]["wv"]]

        return text, source_image, target_image, meta_info

    def get_imgs_via_ids(self, img_ids):
        imgs = []
        for img_id in img_ids:
            imgs.append(self.get_img(self.all_img_names[img_id]))
        return torch.stack(imgs)
