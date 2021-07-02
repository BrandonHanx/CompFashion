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
        self.comp_model = build_composition(
            cfg=cfg, img_channel=self.img_model.out_channels
        )
        self.norm_layer = build_norm_layer(cfg)
        self.loss_func = build_loss_func(cfg)

    def extract_img_feature(self, imgs, single=False):
        if single:
            return self.norm_layer(self.img_model(imgs).mean((2, 3)))
        return self.img_model(imgs).mean((2, 3))

    def extract_text_feature(self, texts, text_lengths):
        return self.text_model(texts, text_lengths)

    def compose_img_text_features(self, img_feats, text_feats):
        return self.norm_layer(self.comp_model(img_feats, text_feats))

    def compose_img_text(self, imgs, texts, text_lengths):
        img_feats = self.extract_img_feature(imgs)
        text_feats = self.extract_text_feature(texts, text_lengths)
        return self.compose_img_text_features(img_feats, text_feats)

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        mod_img1 = self.compose_img_text(imgs_query, mod_texts, text_lengths)
        img2 = self.extract_img_feature(imgs_target, single=True)
        return self.loss_func(mod_img1, img2)


class MultiScaleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_model = build_img_model(cfg)
        self.text_model = build_text_model(cfg)
        self.norm_layer = build_norm_layer(cfg)
        self.loss_func = build_loss_func(cfg)

        self.text_proj_layer = nn.Linear(
            self.text_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.comp_model = nn.ModuleList(
            [
                build_composition(cfg=cfg, img_channel=x)
                for x in self.img_model.out_channels
            ]
        )

    def extract_img_feature(self, imgs, single=False):
        img_feats = self.img_model(imgs)
        if single:
            return [self.norm_layer(x.mean((2, 3))) for x in img_feats]
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj_layer(self.text_model(texts, text_lengths))

    def compose_img_text_features(self, img_feats, text_feats, i):
        return self.norm_layer(self.comp_model[i](img_feats, text_feats))

    def compose_img_text(self, imgs, texts, text_lengths):
        img_feats = self.extract_img_feature(imgs)
        text_feats = self.extract_text_feature(texts, text_lengths)
        return [
            self.compose_img_text_features(x, text_feats, i)
            for i, x in enumerate(img_feats)
        ]

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        mod_img1 = self.compose_img_text(imgs_query, mod_texts, text_lengths)
        img2 = self.extract_img_feature(imgs_target, single=True)
        return self.loss_func(mod_img1, img2)


class ProjModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_proj_layer = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.text_proj_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.text_model.out_channels, cfg.MODEL.COMP.EMBED_DIM),
        )

    def norm_and_avgpool(self, x):
        return self.norm_layer(self.img_proj_layer(x.mean((2, 3))))

    def extract_img_feature(self, imgs, single=False):
        img_feats = self.img_model(imgs)
        if single:
            return self.norm_and_avgpool(img_feats)
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj_layer(self.text_model(texts, text_lengths))

    def compose_img_text_features(self, img_feats, text_feats):
        comp_feats = self.comp_model(img_feats, text_feats)
        return self.norm_and_avgpool(comp_feats)


def build_model(cfg):
    if cfg.MODEL.COMP.METHOD == "base":
        model = Model(cfg)
    elif cfg.MODEL.COMP.METHOD == "proj":
        model = ProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "multi-scale":
        model = MultiScaleModel(cfg)
    else:
        raise NotImplementedError
    return model
