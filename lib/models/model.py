import torch
import torch.nn as nn

from .backbones import build_img_model, build_text_model
from .composition import build_composition
from .correction import build_correction
from .functions import build_loss_func, build_norm_layer

__all__ = ["build_model"]


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_model = build_img_model(cfg)
        self.text_model = build_text_model(cfg)
        self.comp_model = build_composition(
            cfg=cfg,
            img_channel=self.img_model.out_channels,
            text_channel=self.text_model.out_channels,
        )
        self.norm_layer = build_norm_layer(cfg)
        self.loss_func = build_loss_func(cfg)

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_model(imgs).mean((2, 3))
        if norm:
            return self.norm_layer(img_feats)
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_model(texts, text_lengths)

    def compose_img_text_features(self, img_feats, text_feats):
        return self.norm_layer(self.comp_model(img_feats, text_feats))

    def compose_img_text(self, imgs, texts, text_lengths, comp_mode=True):
        img_feats = self.extract_img_feature(imgs)
        text_feats = self.extract_text_feature(texts, text_lengths)
        return self.compose_img_text_features(img_feats, text_feats)

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        mod_img1 = self.compose_img_text(imgs_query, mod_texts, text_lengths)
        img2 = self.extract_img_feature(imgs_target, norm=True)
        return dict(bbc_loss=self.loss_func(mod_img1, img2))


class CombineModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_model = build_img_model(cfg)
        self.text_model = build_text_model(cfg)
        self.comp_model = build_composition(
            cfg=cfg,
            img_channel=self.img_model.out_channels,
            text_channel=self.text_model.out_channels,
        )
        self.outfit_model = build_composition(
            cfg=cfg,
            img_channel=self.img_model.out_channels,
            text_channel=self.text_model.out_channels,
        )
        self.norm_layer = build_norm_layer(cfg)
        self.loss_func = build_loss_func(cfg)

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_model(imgs).mean((2, 3))
        if norm:
            return self.norm_layer(img_feats)
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_model(texts, text_lengths)

    def compose_img_text_features(self, img_feats, text_feats, comp_mode=True):
        if comp_mode:
            return self.norm_layer(self.comp_model(img_feats, text_feats))
        return self.norm_layer(self.outfit_model(img_feats, text_feats))

    def compose_img_text(self, imgs, texts, text_lengths, comp_mode=True):
        img_feats = self.extract_img_feature(imgs, comp_mode)
        text_feats = self.extract_text_feature(texts, text_lengths)
        return self.compose_img_text_features(img_feats, text_feats, comp_mode)

    def compute_loss(
        self,
        imgs_query,
        comp_text,
        comp_text_lengths,
        comp_imgs_target,
        outfit_text,
        outfit_text_lengths,
        outfit_imgs_target,
    ):
        source_img_feat = self.extract_img_feature(imgs_query)
        comp_target_img_feat = self.extract_img_feature(comp_imgs_target, norm=True)
        outfit_target_img_feat = self.extract_img_feature(outfit_imgs_target, norm=True)
        comp_text_feat = self.extract_text_feature(comp_text, comp_text_lengths)
        outfit_text_feat = self.extract_text_feature(outfit_text, outfit_text_lengths)
        comp_feat = self.compose_img_text_features(
            source_img_feat, comp_text_feat, True
        )
        outfit_feat = self.compose_img_text_features(
            source_img_feat, outfit_text_feat, False
        )
        comp_loss = self.loss_func(comp_feat, comp_target_img_feat)
        outfit_loss = self.loss_func(outfit_feat, outfit_target_img_feat)
        return dict(comp_loss=comp_loss, outfit_loss=outfit_loss)


class CombineProjModel(CombineModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.comp_model = build_composition(
            cfg=cfg,
            img_channel=cfg.MODEL.COMP.EMBED_DIM,
            text_channel=self.text_model.out_channels,
        )
        self.outfit_model = build_composition(
            cfg=cfg,
            img_channel=cfg.MODEL.COMP.EMBED_DIM,
            text_channel=self.text_model.out_channels,
        )
        self.comp_proj = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.outfit_proj = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_model(imgs).mean((2, 3))
        if comp_mode:
            img_feats = self.comp_proj(img_feats)
        else:
            img_feats = self.outfit_proj(img_feats)
        if norm:
            return self.norm_layer(img_feats)
        return img_feats

    def compute_loss(
        self,
        imgs_query,
        comp_text,
        comp_text_lengths,
        comp_imgs_target,
        outfit_text,
        outfit_text_lengths,
        outfit_imgs_target,
    ):
        source_img_feat = self.img_model(imgs_query).mean((2, 3))
        comp_target_img_feat = self.extract_img_feature(
            comp_imgs_target, norm=True, comp_mode=True
        )
        outfit_target_img_feat = self.extract_img_feature(
            outfit_imgs_target, norm=True, comp_mode=False
        )
        comp_text_feat = self.extract_text_feature(comp_text, comp_text_lengths)
        outfit_text_feat = self.extract_text_feature(outfit_text, outfit_text_lengths)
        comp_feat = self.compose_img_text_features(
            self.comp_proj(source_img_feat), comp_text_feat, True
        )
        outfit_feat = self.compose_img_text_features(
            self.outfit_proj(source_img_feat), outfit_text_feat, False
        )
        comp_loss = self.loss_func(comp_feat, comp_target_img_feat)
        outfit_loss = self.loss_func(outfit_feat, outfit_target_img_feat)
        return dict(comp_loss=comp_loss, outfit_loss=outfit_loss)


class MultiModel(Model):
    def __init__(self, cfg, k=2):
        super().__init__(cfg)
        self.comp_model_2 = build_composition(
            cfg=cfg,
            img_channel=self.img_model.out_channels,
            text_channel=self.text_model.out_channels,
        )
        self.comp_model = nn.ModuleList([self.comp_model, self.comp_model_2])
        self.combine_fc = nn.Linear(
            self.img_model.out_channels * k, self.img_model.out_channels
        )
        self.k = k

    def compose_img_text_features(self, img_feats, text_feats):
        comp_feats = []
        for i in range(self.k):
            comp_feats.append(self.comp_model[i](img_feats, text_feats))
        comp_feats = comp_feats[0] + comp_feats[1]
        return self.norm_layer(comp_feats)


class MapModel(Model):
    def extract_img_feature(self, imgs, norm=False):
        img_feats = self.img_model(imgs)
        if norm:
            return self.norm_layer(img_feats.mean((2, 3)))
        return img_feats


class DirectModel(Model):
    def extract_img_feature(self, imgs, norm=False):
        img_feats = self.img_model(imgs)
        if norm:
            return self.norm_layer(img_feats)
        return img_feats


class CorrModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.corr_model = build_correction(
            cfg=cfg, img_channel=self.img_model.out_channels
        )

    def diff_img_features(self, ref_img_feats, tar_img_feats):
        return self.norm_layer(self.corr_model(ref_img_feats, tar_img_feats))

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        ref_img_feats = self.extract_img_feature(imgs_query)
        tar_img_feats = self.extract_img_feature(imgs_target)
        text_feats = self.extract_text_feature(mod_texts, text_lengths)

        comp_img_feats = self.compose_img_text_features(ref_img_feats, text_feats)
        corr_text_feats = self.diff_img_features(ref_img_feats, tar_img_feats)

        losses = {}
        losses["comp_bbc"] = self.loss_func(
            comp_img_feats, self.norm_layer(tar_img_feats)
        )
        losses["corr_bbc"] = self.loss_func(
            corr_text_feats, self.norm_layer(text_feats)
        )

        return losses


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
                build_composition(
                    cfg=cfg, img_channel=x, text_channel=self.text_model.out_channels
                )
                for x in self.img_model.out_channels
            ]
        )

    def extract_img_feature(self, imgs, norm=False):
        img_feats = self.img_model(imgs)
        if norm:
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
        img2 = self.extract_img_feature(imgs_target, norm=True)
        return dict(
            bbc_loss=self.loss_func(
                torch.cat(mod_img1, dim=-1), torch.cat(img2, dim=-1)
            )
        )


class ProjModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_proj_layer = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.comp_model = build_composition(
            cfg=cfg,
            img_channel=cfg.MODEL.COMP.EMBED_DIM,
            text_channel=self.text_model.out_channels,
        )

    def extract_img_feature(self, imgs, norm=False):
        img_feats = self.img_proj_layer(self.img_model(imgs).mean((2, 3)))
        if norm:
            return self.norm_layer(img_feats)
        return img_feats


class DIVA(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.comp_model = build_composition(
            cfg=cfg,
            img_channel=cfg.MODEL.COMP.EMBED_DIM,
            text_channel=self.text_model.out_channels,
        )

    def extract_img_feature(self, imgs, norm=False):
        img_feats = self.img_model(imgs)
        if norm:
            img_feats = torch.cat(
                [
                    img_feats["discriminative"],
                    img_feats["selfsimilarity"],
                    img_feats["shared"],
                    img_feats["intra"],
                ],
                dim=1,
            )
            return self.norm_layer(img_feats)
        return img_feats

    def compose_img_text_features(self, img_feats, text_feats):
        return self.norm_layer(
            torch.cat(
                [
                    self.comp_model(img_feats["discriminative"], text_feats),
                    self.comp_model(img_feats["selfsimilarity"], text_feats),
                    self.comp_model(img_feats["shared"], text_feats),
                    self.comp_model(img_feats["intra"], text_feats),
                ],
                dim=1,
            )
        )


def build_model(cfg):
    if cfg.MODEL.COMP.METHOD == "base":
        model = Model(cfg)
    elif cfg.MODEL.COMP.METHOD == "proj":
        model = ProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "multi-scale":
        model = MultiScaleModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "corr":
        model = CorrModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "map":
        model = MapModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "direct":
        model = DirectModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "multi":
        model = MultiModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "diva":
        model = DIVA(cfg)
    elif cfg.MODEL.COMP.METHOD == "combine":
        model = CombineModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "combine-proj":
        model = CombineProjModel(cfg)
    else:
        raise NotImplementedError
    return model
