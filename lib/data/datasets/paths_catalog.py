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
        "fashionpedia_combine_train": {
            "path": "fashionpedia",
            "split": "train",
        },
        "fashionpedia_combine_val": {
            "path": "fashionpedia",
            "split": "val",
        },
    }

    @staticmethod
    def get(name):
        if "fashioniq" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                path=os.path.join(data_dir, attrs["path"]),
                split=attrs["split"],
                cat_type=attrs["cat_type"],
            )
            return dict(
                factory="FashionIQ",
                args=args,
            )
        if "fashionpedia_combine" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                path=os.path.join(data_dir, attrs["path"]),
                split=attrs["split"],
            )
            return dict(
                factory="FashionPediaCombine",
                args=args,
            )
        if "fashionpedia" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                path=os.path.join(data_dir, attrs["path"]),
                split=attrs["split"],
                cat_type=attrs["cat_type"],
            )
            return dict(
                factory="FashionPedia",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
