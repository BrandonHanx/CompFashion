import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_img_model, build_text_model
from .composition import build_composition
from .correction import build_correction
from .criteria import build_loss_func
from .layers import build_norm_layer

__all__ = ["build_model"]


class ConvProjector(nn.Module):
    def __init__(self, in_channel, out_channel):
        assert in_channel > 2 * out_channel
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(),
            nn.Conv2d(
                out_channel * 2,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.model(x)


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
        return {"bbshm": self.loss_func(mod_img1, img2)}  # FIXME


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
        img_feats = self.extract_img_feature(imgs, norm=False, comp_mode=comp_mode)
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
        comp_target_img_feat = self.extract_img_feature(
            comp_imgs_target, norm=True, comp_mode=True
        )
        outfit_target_img_feat = self.extract_img_feature(
            outfit_imgs_target, norm=True, comp_mode=False
        )
        comp_text_feat = self.extract_text_feature(comp_text, comp_text_lengths)
        outfit_text_feat = self.extract_text_feature(outfit_text, outfit_text_lengths)
        comp_feat = self.compose_img_text_features(
            source_img_feat, comp_text_feat, comp_mode=True
        )
        outfit_feat = self.compose_img_text_features(
            source_img_feat, outfit_text_feat, comp_mode=False
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
            text_channel=cfg.MODEL.COMP.EMBED_DIM,
        )
        self.outfit_model = build_composition(
            cfg=cfg,
            img_channel=cfg.MODEL.COMP.EMBED_DIM,
            text_channel=cfg.MODEL.COMP.EMBED_DIM,
        )
        self.comp_proj = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.outfit_proj = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.text_proj = nn.Linear(
            self.text_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
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

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj(self.text_model(texts, text_lengths))

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
            self.comp_proj(source_img_feat), comp_text_feat, comp_mode=True
        )
        outfit_feat = self.compose_img_text_features(
            self.outfit_proj(source_img_feat), outfit_text_feat, comp_mode=False
        )
        comp_loss = self.loss_func(comp_feat, comp_target_img_feat)
        outfit_loss = self.loss_func(outfit_feat, outfit_target_img_feat)
        return dict(comp_loss=comp_loss, outfit_loss=outfit_loss)


class ShareProjCombineProjModel(CombineProjModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.outfit_proj = self.comp_proj


class ShareCompCombineProjModel(CombineProjModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.outfit_model = self.comp_model


class AutoCombineProjModel(CombineProjModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        classifier_dim = cfg.MODEL.COMP.EMBED_DIM
        self.branch_classifier = nn.Sequential(
            nn.Linear(classifier_dim, int(classifier_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(classifier_dim // 2), 2),
        )
        self.vanilla_proj = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_model(imgs).mean((2, 3))
        if norm:
            return self.norm_layer(self.vanilla_proj(img_feats))
        return self.comp_proj(img_feats), self.outfit_proj(img_feats)

    def compose_img_text(self, imgs, texts, text_lengths, comp_mode=True):
        text_feats = self.extract_text_feature(texts, text_lengths)
        weights = self.branch_classifier(text_feats)
        img_feats = self.extract_img_feature(imgs, norm=False)
        return self.selection(img_feats, text_feats, weights)

    def selection(self, img_feats, text_feats, weights):
        weights = F.softmax(weights, -1).detach()
        return weights[:, 0].unsqueeze(-1) * self.compose_img_text_features(
            img_feats[0], text_feats, comp_mode=True
        ) + weights[:, 1].unsqueeze(-1) * self.compose_img_text_features(
            img_feats[1], text_feats, comp_mode=False
        )

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
        source_img_feat = self.extract_img_feature(imgs_query, norm=False)  # tuple
        comp_target_img_feat = self.extract_img_feature(comp_imgs_target, norm=True)
        outfit_target_img_feat = self.extract_img_feature(outfit_imgs_target, norm=True)

        comp_text_feat = self.extract_text_feature(comp_text, comp_text_lengths)
        outfit_text_feat = self.extract_text_feature(outfit_text, outfit_text_lengths)

        comp_weights = self.branch_classifier(comp_text_feat)
        outfit_weights = self.branch_classifier(outfit_text_feat)

        comp_feat = self.selection(source_img_feat, comp_text_feat, comp_weights)
        outfit_feat = self.selection(source_img_feat, outfit_text_feat, outfit_weights)

        comp_loss = self.loss_func(comp_feat, comp_target_img_feat)
        outfit_loss = self.loss_func(outfit_feat, outfit_target_img_feat)
        weights_loss = F.cross_entropy(
            comp_weights, torch.zeros_like(comp_weights[:, 0]).long()
        ) + F.cross_entropy(outfit_weights, torch.ones_like(comp_weights[:, 0]).long())
        return dict(
            comp_loss=comp_loss, outfit_loss=outfit_loss, weights_loss=weights_loss
        )


class HardCombineProjModel(CombineProjModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        classifier_dim = cfg.MODEL.COMP.EMBED_DIM
        self.branch_classifier = nn.Sequential(
            nn.Linear(classifier_dim, int(classifier_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(classifier_dim // 2), 2),
        )

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_model(imgs).mean((2, 3))
        if norm:
            if comp_mode:
                return self.norm_layer(self.comp_proj(img_feats))
            return self.norm_layer(self.outfit_proj(img_feats))
        return self.comp_proj(img_feats), self.outfit_proj(img_feats)

    def compose_img_text(
        self, imgs, texts, text_lengths, comp_mode=True, return_weights=False
    ):
        text_feats = self.extract_text_feature(texts, text_lengths)
        if isinstance(imgs, tuple):
            img_feats = imgs
        else:
            img_feats = self.extract_img_feature(imgs, norm=False)  # tuple
        return self.selection(img_feats, text_feats, return_weights)

    def selection(self, img_feats, text_feats, return_weights):
        weights = self.branch_classifier(text_feats).detach()
        weights = (F.softmax(weights, -1) > 0.5).float()  # threshold for hard selection
        selected_feats = weights[:, 0].unsqueeze(-1) * self.compose_img_text_features(
            img_feats[0], text_feats, comp_mode=True
        ) + weights[:, 1].unsqueeze(-1) * self.compose_img_text_features(
            img_feats[1], text_feats, comp_mode=False
        )
        if return_weights:
            return selected_feats, weights
        return selected_feats

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
        comp_proj_feat, outfit_proj_feat = self.extract_img_feature(
            imgs_query, norm=False
        )  # tuple
        comp_target_img_feat = self.extract_img_feature(
            comp_imgs_target, norm=True, comp_mode=True
        )
        outfit_target_img_feat = self.extract_img_feature(
            outfit_imgs_target, norm=True, comp_mode=False
        )

        comp_text_feat = self.extract_text_feature(comp_text, comp_text_lengths)
        outfit_text_feat = self.extract_text_feature(outfit_text, outfit_text_lengths)

        comp_weights = self.branch_classifier(comp_text_feat)
        outfit_weights = self.branch_classifier(outfit_text_feat)

        comp_feat = self.compose_img_text_features(
            comp_proj_feat, comp_text_feat, comp_mode=True
        )
        outfit_feat = self.compose_img_text_features(
            outfit_proj_feat, outfit_text_feat, comp_mode=False
        )

        comp_loss = self.loss_func(comp_feat, comp_target_img_feat)
        outfit_loss = self.loss_func(outfit_feat, outfit_target_img_feat)
        weights_loss = F.cross_entropy(
            comp_weights, torch.zeros_like(comp_weights[:, 0]).long()
        ) + F.cross_entropy(outfit_weights, torch.ones_like(comp_weights[:, 0]).long())
        return dict(
            comp_loss=comp_loss, outfit_loss=outfit_loss, weights_loss=weights_loss
        )


class HardCombineProjMapModel(HardCombineProjModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.comp_proj = ConvProjector(
            self.img_model.out_channels,
            cfg.MODEL.COMP.EMBED_DIM,
        )
        self.outfit_proj = ConvProjector(
            self.img_model.out_channels,
            cfg.MODEL.COMP.EMBED_DIM,
        )

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_model(imgs)
        if norm:
            if comp_mode:
                return self.norm_layer(self.comp_proj(img_feats).mean((2, 3)))
            return self.norm_layer(self.outfit_proj(img_feats).mean((2, 3)))
        return self.comp_proj(img_feats), self.outfit_proj(img_feats)

    def compose_img_text_features(self, img_feats, text_feats, comp_mode=True):
        if comp_mode:
            return self.norm_layer(self.comp_model(img_feats, text_feats).mean((2, 3)))
        return self.norm_layer(self.outfit_model(img_feats, text_feats).mean((2, 3)))


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
                    cfg=cfg, img_channel=x, text_channel=cfg.MODEL.COMP.EMBED_DIM
                )
                for x in self.img_model.out_channels
            ]
        )

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
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

    def compose_img_text(self, imgs, texts, text_lengths, comp_mode=True):
        img_feats = self.extract_img_feature(imgs)
        text_feats = self.extract_text_feature(texts, text_lengths)
        img_feats = [
            self.compose_img_text_features(x, text_feats, i)
            for i, x in enumerate(img_feats)
        ]
        return torch.cat(img_feats, dim=-1)

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        mod_img1 = self.compose_img_text(imgs_query, mod_texts, text_lengths)
        img2 = self.extract_img_feature(imgs_target, norm=True)
        return dict(bbc_loss=self.loss_func(mod_img1, img2))


class ProjModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_proj_layer = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.text_proj_layer = nn.Linear(
            self.text_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.comp_model = build_composition(
            cfg=cfg,
            img_channel=cfg.MODEL.COMP.EMBED_DIM,
            text_channel=cfg.MODEL.COMP.EMBED_DIM,
        )

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_proj_layer(self.img_model(imgs).mean((2, 3)))
        if norm:
            return self.norm_layer(img_feats)
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj_layer(self.text_model(texts, text_lengths))


class ProjMapModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.img_proj_layer = nn.Linear(
        #     self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        # )
        self.img_proj_layer = ConvProjector(
            self.img_model.out_channels,
            cfg.MODEL.COMP.EMBED_DIM,
        )
        self.text_proj_layer = nn.Linear(
            self.text_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.comp_model = build_composition(
            cfg=cfg,
            img_channel=cfg.MODEL.COMP.EMBED_DIM,
            text_channel=cfg.MODEL.COMP.EMBED_DIM,
        )

    def extract_img_feature(self, imgs, norm=False, comp_mode=True):
        img_feats = self.img_proj_layer(self.img_model(imgs))
        if norm:
            return self.norm_layer(img_feats.mean((2, 3)))
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj_layer(self.text_model(texts, text_lengths))

    def compose_img_text_features(self, img_feats, text_feats):
        comp_feats = self.comp_model(img_feats, text_feats)
        return self.norm_layer(comp_feats.mean((2, 3)))


def build_model(cfg):
    if cfg.MODEL.COMP.METHOD == "base":
        model = Model(cfg)
    elif cfg.MODEL.COMP.METHOD == "proj":
        model = ProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "proj-map":
        model = ProjMapModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "multi-scale":
        model = MultiScaleModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "corr":
        model = CorrModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "combine":
        model = CombineModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "combine-proj":
        model = CombineProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "shareproj-combine-proj":
        model = ShareProjCombineProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "sharecomp-combine-proj":
        model = ShareCompCombineProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "auto-combine-proj":
        model = AutoCombineProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "hard-combine-proj":
        model = HardCombineProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "hard-combine-proj-map":
        model = HardCombineProjMapModel(cfg)
    else:
        raise NotImplementedError
    return model
