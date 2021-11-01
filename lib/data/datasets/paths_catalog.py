import os


class DatasetCatalog:
    DATA_DIR = "train_data"
    DATASETS = {
        "fashioniq_dress_train": {
            "path": "fashioniq",
            "split": "train",
            "cat_type": "dress",
        },
        "fashioniq_shirt_train": {
            "path": "fashioniq",
            "split": "train",
            "cat_type": "shirt",
        },
        "fashioniq_toptee_train": {
            "path": "fashioniq",
            "split": "train",
            "cat_type": "toptee",
        },
        "fashioniq_dress_val": {
            "path": "fashioniq",
            "split": "val",
            "cat_type": "dress",
        },
        "fashioniq_shirt_val": {
            "path": "fashioniq",
            "split": "val",
            "cat_type": "shirt",
        },
        "fashioniq_toptee_val": {
            "path": "fashioniq",
            "split": "val",
            "cat_type": "toptee",
        },
        "fashionpedia_comp_train": {
            "path": "fashionpedia",
            "split": "train",
            "cat_type": "comp",
        },
        "fashionpedia_comp_val": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "comp",
        },
        "fashionpedia_comp_test": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "comp",
        },
        "fashionpedia_comp_miner_train": {
            "path": "fashionpedia",
            "split": "train",
            "cat_type": "comp_miner",
        },
        "fashionpedia_comp_miner_val": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "comp_miner",
        },
        "fashionpedia_comp_miner_test": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "comp_miner",
        },
        "fashionpedia_outfit_train": {
            "path": "fashionpedia",
            "split": "train",
            "cat_type": "outfit",
        },
        "fashionpedia_outfit_val": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "outfit",
        },
        "fashionpedia_outfit_test": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "outfit",
        },
        "fashionpedia_outfit_wa_train": {
            "path": "fashionpedia",
            "split": "train",
            "cat_type": "outfit_wa",
        },
        "fashionpedia_outfit_wa_val": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "outfit_wa",
        },
        "fashionpedia_outfit_wa_test": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "outfit_wa",
        },
        "fashionpedia_combine_train": {
            "path": "fashionpedia",
            "split": "train",
        },
        "fashionpedia_combine_val": {
            "path": "fashionpedia",
            "split": "val",
        },
        "fashionpedia_comp_val_turn3": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "comp",
            "turn": 3,
        },
        "fashionpedia_comp_test_turn3": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "comp",
            "turn": 3,
        },
        "fashionpedia_comp_val_turn5": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "comp",
            "turn": 5,
        },
        "fashionpedia_comp_test_turn5": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "comp",
            "turn": 5,
        },
        "fashionpedia_hybrid_val_turn3": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "hybrid",
            "turn": 3,
        },
        "fashionpedia_hybrid_test_turn3": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "hybrid",
            "turn": 3,
        },
        "fashionpedia_hybrid_val_turn5": {
            "path": "fashionpedia",
            "split": "val",
            "cat_type": "hybrid",
            "turn": 5,
        },
        "fashionpedia_hybrid_test_turn5": {
            "path": "fashionpedia",
            "split": "test",
            "cat_type": "hybrid",
            "turn": 5,
        },
    }

    @staticmethod
    def get(name):
        def get_attrs(name):
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            attrs["path"] = os.path.join(data_dir, attrs["path"])
            return attrs

        if "fashioniq" in name:
            return dict(
                factory="FashionIQ",
                args=get_attrs(name),
            )
        if "fashionpedia_combine" in name:
            return dict(
                factory="FashionPediaCombine",
                args=get_attrs(name),
            )
        if "fashionpedia" in name:
            if "turn" in name:
                return dict(factory="FashionPediaMultiTurn", args=get_attrs(name))
            return dict(
                factory="FashionPedia",
                args=get_attrs(name),
            )
        raise RuntimeError("Dataset not available: {}".format(name))
