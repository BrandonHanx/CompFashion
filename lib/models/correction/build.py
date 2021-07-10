from .fd import build_fd


def build_correction(**kwargs):
    cfg = kwargs["cfg"]
    if cfg.MODEL.CORR_MODEL == "fd":
        model = build_fd(cfg, kwargs["img_channel"])
    else:
        raise NotImplementedError
    return model
