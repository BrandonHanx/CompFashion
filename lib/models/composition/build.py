from .cosmo import build_cosmo
from .mmt import build_mmt
from .rtic import build_rtic
from .tirg import build_tirg
from .val import build_val


def build_composition(**kwargs):
    cfg = kwargs["cfg"]
    if cfg.MODEL.COMP_MODEL == "tirg":
        model = build_tirg(cfg)
    elif cfg.MODEL.COMP_MODEL == "val":
        model = build_val(cfg, kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "rtic":
        model = build_rtic(cfg, kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "cosmo":
        model = build_cosmo(cfg)
    elif cfg.MODEL.COMP_MODEL == "mmt":
        model = build_mmt(cfg)
    else:
        raise NotImplementedError
    return model
