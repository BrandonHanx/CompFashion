import math

import torch
import torch.nn as nn

from .backbones import build_img_model, build_text_model
from .composition import build_composition
from .correction import build_correction
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
        return dict(bbc_loss=self.loss_func(mod_img1, img2))


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


class CorrBmmModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.corr_model = build_correction(
            cfg=cfg, img_channel=self.img_model.out_channels
        )

    def diff_img_features(self, ref_img_feats, tar_img_feats):
        N = tar_img_feats.shape[0]
        corr_text_feats = []
        for ref_img_feat in ref_img_feats:
            corr_text_feats.append(
                self.corr_model(ref_img_feat.unsqueeze(0).expand(N, -1), tar_img_feats)
            )  # N x D
        corr_text_feats = torch.stack(corr_text_feats, dim=0)  # B x N x D
        return self.norm_layer(corr_text_feats)

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        ref_img_feats = self.extract_img_feature(imgs_query)
        tar_img_feats = self.extract_img_feature(imgs_target)
        text_feats = self.extract_text_feature(mod_texts, text_lengths)  # B x D

        comp_img_feats = self.compose_img_text_features(ref_img_feats, text_feats)
        corr_text_feats = self.diff_img_features(
            ref_img_feats, tar_img_feats
        )  # B x N x D

        losses = {}
        losses["comp_bbc"] = self.loss_func(
            comp_img_feats, self.norm_layer(tar_img_feats)
        )
        losses["corr_bbc"] = self.loss_func(
            corr_text_feats, self.norm_layer(text_feats), bmm=True
        )

        return losses


class CorrCycleModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.corr_model = build_correction(
            cfg=cfg, img_channel=self.img_model.out_channels
        )

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        ref_img_feats = self.extract_img_feature(imgs_query)
        tar_img_feats = self.extract_img_feature(imgs_target)
        text_feats = self.extract_text_feature(mod_texts, text_lengths)

        comp_img_feats = self.comp_model(ref_img_feats, text_feats)
        corr_text_feats = self.corr_model(ref_img_feats, tar_img_feats)

        fake_text_feats = self.corr_model(tar_img_feats, ref_img_feats)
        cycle_img_feats = self.comp_model(tar_img_feats, fake_text_feats.detach())

        losses = {}
        losses["comp_bbc"] = self.loss_func(
            self.norm_layer(comp_img_feats), self.norm_layer(tar_img_feats)
        )
        losses["corr_bbc"] = self.loss_func(
            self.norm_layer(corr_text_feats), self.norm_layer(text_feats)
        )
        losses["cycle_comp_bbc"] = self.loss_func(
            self.norm_layer(cycle_img_feats), self.norm_layer(ref_img_feats)
        )

        return losses


class AttnPoolModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
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
        self.img_proj_layer = nn.Linear(
            self.img_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )
        self.text_proj_layer = nn.Linear(
            self.text_model.out_channels, cfg.MODEL.COMP.EMBED_DIM
        )

    def extract_img_feature(self, imgs, single=False):
        img_feats = (
            self.img_model(imgs).flatten(start_dim=-2, end_dim=-1).transpose(-2, -1)
        )
        img_feats = self.img_proj_layer(img_feats)
        if single:
            return self.norm_layer(img_feats.mean(-2))
        return img_feats

    def extract_text_feature(self, texts, text_lengths):
        return self.text_proj_layer(self.text_model(texts, text_lengths))

    def compose_img_text_features(self, img_feats, text_feats):
        return self.norm_layer(self.comp_model(img_feats, text_feats).mean(1))

    def compose_img_text(self, imgs, texts, text_lengths):
        img_feats = self.extract_img_feature(imgs)
        text_feats = self.extract_text_feature(texts, text_lengths)
        return dict(bbc_loss=self.compose_img_text_features(img_feats, text_feats))


class ClusterLoss(nn.Module):
    def __init__(self, class_num=64, temperature=1.0, device="cuda"):
        super().__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    @staticmethod
    def mask_correlated_clusters(class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return {"cluster_loss": loss + ne_loss}


class TransClusterModel(TransModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cluster_proj = nn.Sequential(
            nn.Linear(cfg.MODEL.COMP.EMBED_DIM, cfg.MODEL.COMP.EMBED_DIM),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.COMP.EMBED_DIM, 64),
            nn.Softmax(dim=1),
        )
        self.cluster_loss = ClusterLoss()

    def compute_loss(self, imgs_query, mod_texts, text_lengths, imgs_target):
        text_feat = self.extract_text_feature(mod_texts, text_lengths)
        img_feat_q = self.extract_img_feature(imgs_query)
        img_feat_t = self.extract_img_feature(imgs_target)
        comp_feat = self.comp_model(img_feat_q, text_feat)

        comp_feat_c = self.cluster_proj(self.norm_layer(comp_feat.flatten(0, 1)))
        img_feat_t_c = self.cluster_proj(self.norm_layer(img_feat_t.flatten(0, 1)))

        losses = {}
        losses["bbc_loss"] = self.loss_func(
            self.norm_layer(comp_feat.mean(1)), self.norm_layer(img_feat_t.mean(1))
        )
        losses.update(self.cluster_loss(comp_feat_c, img_feat_t_c))

        return losses


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
    elif cfg.MODEL.COMP.METHOD == "trans-cluster":
        model = TransClusterModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "corr":
        model = CorrModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "corr-cycle":
        model = CorrCycleModel(cfg)
    elif cfg.MODEL.COMP.METHOD == "corr-bmm":
        model = CorrBmmModel(cfg)
    else:
        raise NotImplementedError
    return model
