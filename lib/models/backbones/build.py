from .bigru import build_bigru
from .fc import build_fc
from .lstm import build_lstm
from .m_resnet import build_m_resnet
from .resnet import build_resnet


def build_img_model(cfg):
    if cfg.MODEL.IMG_MODEL in ["resnet18", "resnet50", "resnet101"]:
        model = build_resnet(cfg)
    elif cfg.MODEL.IMG_MODEL in ["m_resnet50", "m_resnet101"]:
        model = build_m_resnet(cfg)
    else:
        raise NotImplementedError
    return model


def build_text_model(cfg):
    if cfg.MODEL.TEXT_MODEL == "bigru":
        model = build_bigru(cfg)
    elif cfg.MODEL.TEXT_MODEL == "lstm":
        model = build_lstm(cfg)
    #     elif cfg.MODEL.TEXT_MODEL == "bert":
    #         model = build_bert(cfg)
    elif cfg.MODEL.TEXT_MODEL == "fc":
        model = build_fc(cfg)
    else:
        raise NotImplementedError
    return model
