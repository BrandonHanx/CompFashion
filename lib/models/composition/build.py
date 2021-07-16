from .cosmo import build_cosmo
from .mmt import build_mmt
from .rtic import build_rtic
from .tirg import build_tirg
from .trans import build_trans
from .transdec import build_transdec
from .transenc import build_transenc
from .val import build_val
from .xcit import build_xcit


def build_composition(**kwargs):
    cfg = kwargs["cfg"]
    if cfg.MODEL.COMP_MODEL == "tirg":
        model = build_tirg(cfg, kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "val":
        model = build_val(cfg, kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "rtic":
        model = build_rtic(cfg, kwargs["img_channel"])
    elif cfg.MODEL.COMP_MODEL == "cosmo":
        model = build_cosmo(cfg)
    elif cfg.MODEL.COMP_MODEL == "mmt":
        model = build_mmt(cfg)
    elif cfg.MODEL.COMP_MODEL == "xcit":
        model = build_xcit(cfg)
    elif cfg.MODEL.COMP_MODEL == "trans":
        model = build_trans(cfg)
    elif cfg.MODEL.COMP_MODEL == "transdec":
        model = build_transdec(cfg)
    elif cfg.MODEL.COMP_MODEL == "transenc":
        model = build_transenc(cfg)
    else:
        raise NotImplementedError
    return model
