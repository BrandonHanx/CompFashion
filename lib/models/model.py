import torch.nn as nn

from .backbones import build_img_model, build_text_model
from .composition import build_composition
from .functions import build_loss_func, build_norm_layer

__all__ = ["build_model"]


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_model = build_img_model(cfg)
        self.text_model = build_text_model(cfg)
        self.comp_model = build_composition(cfg)
        self.norm_layer = build_norm_layer(cfg)
        self.loss_func = build_loss_func(cfg)

    def extract_img_feature(self, imgs):
        return self.img_model(imgs)

    def extract_text_feature(self, texts, text_lengths):
        return self.text_model(texts, text_lengths)

    def compose_img_text_features(self, img_feats, text_feats):
        return self.comp_model(img_feats, text_feats)

    def compose_img_text(self, imgs, texts, text_lengths):
        img_feats = self.extract_img_feature(imgs)
        text_feats = self.extract_text_feature(texts, text_lengths)
        return self.compose_img_text_features(img_feats, text_feats)

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        mod_img1 = self.compose_img_text(imgs_query, mod_texts, text_lengths)
        mod_img1 = self.norm_layer(mod_img1)
        img2 = self.extract_img_feature(imgs_target)
        img2 = self.norm_layer(img2)
        assert mod_img1.shape[0] == img2.shape[0] and mod_img1.shape[1] == img2.shape[1]
        return self.loss_func(mod_img1, img2)


def build_model(cfg):
    return Model(cfg)
