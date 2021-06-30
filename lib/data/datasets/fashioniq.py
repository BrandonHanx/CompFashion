import PIL
import torch
import torch.utils.data

from lib.utils.directory import memcache, np_loader, read_json


class FashionIQ(torch.utils.data.Dataset):
    """FashionIQ dataset."""

    def __init__(
        self, path, split="train", cat_type="dress", transform=None, vocab="glove"
    ):
        super().__init__()
        self.path = path
        self.transform = transform
        self.split = split
        self.data = []
        self.name = f"FashionIQ.{cat_type}.dict.{split}"

        caps_file = f"{path}/captions/cap.{cat_type}.dict.{split}.pkl"
        split_file = f"{path}/image_splits/split.{cat_type}.{split}.json"
        vocab_file = f"{path}/captions/{vocab}.npy"

        self.data = memcache(caps_file)
        self.vocab = np_loader(vocab_file)
        self.all_img_names = read_json(split_file)
        self.all_img_ids = {
            self.all_img_names[x]: x for x in range(len(self.all_img_names))
        }

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
        text = self.vocab[self.data[idx]["wv"]]

        return text, source_image, target_image, meta_info

    def __len__(self):
        return len(self.data)

    def get_img(self, img_name):
        img_path = f"{self.path}/images/{img_name}.jpg"

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
