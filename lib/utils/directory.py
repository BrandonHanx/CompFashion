import json
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def pickle_loader(pkl_path):
    tic = time.time()
    print("loading features from {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    print("done in {:.3f}s".format(time.time() - tic))
    return data


def np_loader(np_path, l2norm=False):
    tic = time.time()
    print("loading features from {}".format(np_path))
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1", allow_pickle=True)
    print("done in {:.3f}s".format(time.time() - tic))
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1e-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1e-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data


def memcache(path):
    suffix = Path(path).suffix
    if suffix in {".pkl", ".pickle"}:
        res = pickle_loader(path)
    elif suffix == ".npy":
        res = np_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix}")
    return res


def read_json(fname):
    with open(fname, "rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with open(fname, "wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
