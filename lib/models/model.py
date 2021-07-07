import torch
import torch.nn as nn

from .backbones import build_img_model, build_text_model
from .composition import build_composition
from .functions import build_attn_pool, build_loss_func, build_norm_layer

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
        img_feats = self.img_model(imgs).mean((2, 3))
        if single:
            return self.norm_layer(img_feats)
        return img_feats

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


class AttnPoolModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_pool = build_attn_pool(cfg)

    def extract_img_feature(self, imgs, single=False):
        img_feats = self.attn_pool(self.img_model(imgs))
        if single:
            return self.norm_layer(img_feats)
        return img_feats


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
            return torch.cat(
                [self.norm_layer(x.mean((2, 3))) for x in img_feats], dim=-1
            )
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
        return self.loss_func(torch.cat(mod_img1, dim=-1), torch.cat(img2, dim=-1))


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


class TransModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_proj_layer = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.text_proj_layer = nn.Linear(
            cfg.MODEL.GRU.VOCABULARY_SIZE, cfg.MODEL.COMP.EMBED_DIM
        )
        self.pseudoclass_layer = nn.Linear(
            cfg.MODEL.COMP.EMBED_DIM, cfg.MODEL.COMP.EMBED_DIM
        )

    def extract_img_feature(self, imgs, single=False):
        img_feats = (
            self.img_model(imgs).flatten(start_dim=-2, end_dim=-1).transpose(-2, -1)
        )
        img_feats = self.img_proj_layer(img_feats)
        if single:
            return self.norm_layer(img_feats.mean(-2))
        return img_feats

    def extract_text_feature(self, texts):
        return self.text_proj_layer(self.text_model(texts))

    def compose_img_text_features(self, img_feats, text_feats, text_lengths):
        return self.norm_layer(
            self.comp_model(img_feats, text_feats, text_lengths).mean(-2)
        )

    def compose_img_text(self, imgs, texts, text_lengths):
        img_feats = self.extract_img_feature(imgs)
        text_feats = self.extract_text_feature(texts)
        return self.compose_img_text_features(img_feats, text_feats, text_lengths)

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        img_feats = self.extract_img_feature(imgs_query)
        text_feats = self.extract_text_feature(mod_texts)
        mod_patches = self.comp_model(img_feats, text_feats, text_lengths)
        mod_img_feats = self.norm_layer(mod_patches.mean(-2))
        img_feats_2 = self.extract_img_feature(imgs_target)
        tar_img_feats = self.norm_layer(img_feats_2.mean(-2))

        loss = {}
        loss.update(self.loss_func[0](mod_img_feats, tar_img_feats))
        loss.update(
            self.loss_func[1](
                self.pseudoclass_layer(mod_patches.flatten(0, 1)),
                self.pseudoclass_layer(img_feats_2.flatten(0, 1)),
            )
        )


def build_model(cfg):
    if cfg.MODEL.COMP.METHOD == "base":
        model = Model(cfg)
    elif cfg.MODEL.COMP.METHOD == "proj":
        model = ProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "multi-scale":
        model = MultiScaleModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "trans":
        model = TransModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "attn-pool":
        model = AttnPoolModel(cfg)
    else:
        raise NotImplementedError
    return model
