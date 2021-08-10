import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return dict(bbc_loss=self.loss_func(mod_img1, img2))


class MapModel(Model):
    def extract_img_feature(self, imgs, single=False):
        img_feats = self.img_model(imgs)
        if single:
            return self.norm_layer(img_feats.mean((2, 3)))
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
        embed_dim = cfg.MODEL.COMP.EMBED_DIM
        self.img_proj_layer = nn.Sequential(
            nn.Linear(self.img_model.out_channels, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.text_proj_layer = nn.Sequential(
            nn.Linear(self.text_model.out_channels, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def extract_img_feature(self, imgs, single=False):
        img_feats = (
            self.img_model(imgs).flatten(start_dim=-2, end_dim=-1).transpose(-2, -1)
        )
        img_feats = self.img_proj_layer(img_feats)
        if single:
            img_feats = self.comp_model(img_feats)
            return self.norm_layer(img_feats)
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj_layer(self.text_model(texts, text_lengths))


class TransDecModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.MODEL.COMP.EMBED_DIM

        self.img_model = build_img_model(cfg).eval()
        self.img_model.train = self._disabled_train

        self.comp_model = build_composition(cfg=cfg)
        self.text_model = build_text_model(cfg)
        self.text_proj_layer = nn.Sequential(
            nn.Linear(self.text_model.out_channels, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def _disabled_train(self, mode=True):
        return self

    def extract_img_feature(self, imgs):
        imgs = imgs * 2 - 1
        _, indices = self.img_model.encode(imgs)
        return indices.flatten(1, -1)

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj_layer(self.text_model(texts, text_lengths))

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        ref_indices = self.extract_img_feature(imgs_query).long()
        tgt_indices = self.extract_img_feature(imgs_target).long()
        text_feat = self.extract_text_feature(mod_texts, text_lengths)

        pred_logits = self.comp_model(ref_indices, text_feat, tgt_indices)
        loss = dict(
            ce=F.cross_entropy(
                pred_logits.view(-1, pred_logits.size(-1)), tgt_indices.view(-1)
            )
        )
        return loss

    @torch.no_grad()
    def reconstruct(self, imgs_query, mod_texts, text_lengths, imgs_target):
        ref_indices = self.extract_img_feature(imgs_query).long()
        tgt_indices = self.extract_img_feature(imgs_target).long()
        text_feat = self.extract_text_feature(mod_texts, text_lengths)

        pred_indices = self.comp_model.sample(
            ref_indices.flatten(1, -1), text_feat, tgt_indices.flatten(1, -1)
        )
        tgt_img = self.img_model.decode_code(tgt_indices.view(-1, 16, 16))
        pred_img = self.img_model.decode_code(pred_indices.view(-1, 16, 16))
        return self.to_rgb(pred_img), self.to_rgb(tgt_img)

    @staticmethod
    def to_rgb(img):
        img = (img + 1.0) / 2.0
        return img.clamp(0, 1)


def build_model(cfg):
    if cfg.MODEL.COMP.METHOD == "base":
        model = Model(cfg)
    elif cfg.MODEL.COMP.METHOD == "proj":
        model = ProjModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "multi-scale":
        model = MultiScaleModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "trans":
        model = TransModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "transdec":
        model = TransDecModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "corr":
        model = CorrModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "map":
        model = MapModel(cfg)
    else:
        raise NotImplementedError
    return model
