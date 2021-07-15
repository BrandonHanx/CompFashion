import torch.nn as nn

from .bert import build_bert
from .bigru import build_bigru
from .lstm import build_lstm
from .resnet import build_resnet
from .vqgan import build_vqgan


def build_img_model(cfg):
    if cfg.MODEL.IMG_MODEL in ["resnet18", "resnet50", "resnet101"]:
        model = build_resnet(cfg)
    elif cfg.MODEL.IMG_MODEL == "vqgan":  # FIXME: adapt to 8196
        model = build_vqgan()
    else:
        raise NotImplementedError
    return model


def build_text_model(cfg):
    if cfg.MODEL.TEXT_MODEL == "bigru":
        model = build_bigru(cfg)
    elif cfg.MODEL.TEXT_MODEL == "lstm":
        model = build_lstm(cfg)
    elif cfg.MODEL.TEXT_MODEL == "bert":
        model = build_bert(cfg)
    elif cfg.MODEL.TEXT_MODEL == "none":
        model = nn.Identity()
    else:
        raise NotImplementedError
    return model
