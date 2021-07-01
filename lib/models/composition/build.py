from .cosmo import build_cosmo
from .tirg import build_tirg
from .val import build_val


def build_composition(**kwargs):
    cfg = kwargs["cfg"]
    if cfg.MODEL.COMP_MODEL == "tirg":
        model = build_tirg(cfg)
    elif cfg.MODEL.COMP_MODEL == "val":
        model = build_val(cfg, kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "cosmo":
        model = build_cosmo(cfg)
    else:
        raise NotImplementedError
    return model
