from .bert import build_bert
from .bigru import build_bigru
from .resnet import build_resnet


def build_img_model(cfg):
    if cfg.MODEL.IMG_MODEL in ["resnet18", "resnet50", "resnet101"]:
        model = build_resnet(cfg)
    else:
        raise NotImplementedError
    return model


def build_text_model(cfg):
    if cfg.MODEL.TEXT_MODEL == "bigru":
        model = build_bigru(cfg)
    elif cfg.MODEL.TEXT_MODEL == "bert":
        model = build_bert(cfg)
    else:
        raise NotImplementedError
    return model
