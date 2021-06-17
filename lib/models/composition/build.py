from .simple_head.head import build_simple_head


def build_composition(cfg, visual_out_channels, textual_out_channels):
    return build_simple_head(cfg, visual_out_channels, textual_out_channels)
