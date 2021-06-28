from .tirg import build_tirg


def build_composition(cfg):
    if cfg.MODEL.COMP_MODEL == "tirg":
        model = build_tirg(cfg)
    else:
        raise NotImplementedError
    return model
