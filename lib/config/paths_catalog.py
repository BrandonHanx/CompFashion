# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

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
        raise RuntimeError("Dataset not available: {}".format(name))
