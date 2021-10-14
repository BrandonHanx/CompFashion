from .cosmo import build_cosmo
from .csa_net import build_csanet
from .rtic import build_rtic
from .tirg import build_tirg
from .val import build_val


def build_composition(**kwargs):
    cfg = kwargs["cfg"]
    if cfg.MODEL.COMP_MODEL == "tirg":
        model = build_tirg(kwargs["text_channel"], kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "val":
        model = build_val(kwargs["text_channel"], kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "rtic":
        model = build_rtic(kwargs["text_channel"], kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "csa-net":
        model = build_csanet(kwargs["text_channel"], kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "cosmo":
        model = build_cosmo(kwargs["text_channel"], kwargs["img_channel"])
    else:
        raise NotImplementedError
    return model
