import numpy as np

from lib.data.collate_batch import multiturn_collate_fn

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
        self.turn = turn
        super().__init__(path, split, cat_type, transform, vocab)

    def get_name(self):
        return f"FashionPedia.{self.cat_type}.dict.{self.split}.turn{self.turn}"

    def get_file_name(self):
        return f"{self.path}/{self.cat_type}_triplets_dict_{self.split}_turn{self.turn}.json"

    def __getitem__(self, idx):
        # FIXME: need optimization
        source_img_name = self.data[idx]["candidate"]
        target_img_name = self.data[idx]["target"]
        meta_info = {
            "source_img_id": self.all_img_ids[source_img_name],
            "target_img_id": self.all_img_ids[target_img_name],
            "turn_idxs": self.data[idx]["turn_idxs"],
        }
        source_image = self.get_img(source_img_name)
        target_image = self.get_img(target_img_name)
        if self.vocab_type == "init":
            text = [np.array(x) for x in self.data[idx]["wv"]]
        elif self.vocab_type == "two-hot":
            raise NotImplementedError
        else:
            text = [self.vocab[x] for x in self.data[idx]["wv"]]

        args = [source_image, target_image, meta_info]
        args.extend(text)

        return tuple(args)

    def get_specific_turn(self, turn_mode, batch_size=64):
        turn_mode_list = []
        for i, data in enumerate(self.data):
            if data["turn_mode"] == turn_mode:
                turn_mode_list.append(i)

        data = []
        for idx in turn_mode_list:
            data.append(self.__getitem__(idx))
            if len(data) == batch_size or idx == turn_mode_list[-1]:
                batch_data = data
                data = []
                yield multiturn_collate_fn(batch_data)
