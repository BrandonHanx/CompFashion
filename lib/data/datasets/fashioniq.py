import PIL
import torch
import torch.utils.data

from lib.utils.directory import memcache


class FashionIQ(torch.utils.data.Dataset):
    """FashionIQ dataset."""

    def __init__(
        self, path, split="train", cat_type="dress", transform=None, text_feat="glove"
    ):
        super().__init__()
        self.path = path
        self.transform = transform
        self.split = split
        self.data = []

        caps_file = f"{path}/captions/cap.{cat_type}.{text_feat}.{split}.pkl"
        self.data = memcache(caps_file)

    def __getitem__(self, idx):
        source_img_id = self.data[idx]["candidate"]
        target_img_id = self.data[idx]["target"]
        meta_info = {
            "source_img_id": source_img_id,
            "target_img_id": target_img_id,
            "original_captions": " ".join(self.data[idx]["captions"]).lower(),
        }
        source_image = self.get_img(source_img_id)
        target_image = self.get_img(target_img_id)
        text = self.data[idx]["wv"]

        return text, source_image, target_image, meta_info

    def __len__(self):
        return len(self.data)

    def get_img(self, img_id):
        img_path = f"{self.path}/all_imgs/{img_id}.jpg"

        with open(img_path, "rb") as f:
            img = PIL.Image.open(f)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
