import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_norm_layer", "build_loss_func"]


class NormalizationLayer(nn.Module):
    """Class for normalization layer."""

    def __init__(self, normalize_scale=4.0, learn_scale=True):
        super().__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        return features


class BatchBasedClassificationLoss(nn.Module):
    @staticmethod
    def forward(ref_features, tar_features):
        batch_size = ref_features.size(0)
        device = ref_features.device

        pred = ref_features.mm(tar_features.transpose(0, 1))
        labels = torch.arange(0, batch_size).long().to(device)
        loss = {"batch_based_classification_loss": F.cross_entropy(pred, labels)}
        return loss


class MultiScaleBatchBasedClassificationLoss(nn.Module):
    #     @staticmethod
    #     def forward(ms_ref_features, ms_tar_features):
    #         batch_size = ms_ref_features[0].size(0)
    #         device = ms_ref_features[0].device

    #         loss = 0.0
    #         for ref_features, tar_features in zip(ms_ref_features, ms_tar_features):
    #             pred = ref_features.mm(tar_features.transpose(0, 1))
    #             labels = torch.arange(0, batch_size).long().to(device)
    #             loss = loss + F.cross_entropy(pred, labels)

    #         loss = {"batch_based_classification_loss": loss / len(ms_ref_features)}
    #         return loss
    @staticmethod
    def forward(ref_features, tar_features):
        ref_features = torch.cat(ref_features, dim=-1)
        tar_features = torch.cat(tar_features, dim=-1)
        batch_size = ref_features.size(0)
        device = ref_features.device

        pred = ref_features.mm(tar_features.transpose(0, 1))
        labels = torch.arange(0, batch_size).long().to(device)
        loss = {"batch_based_classification_loss": F.cross_entropy(pred, labels)}
        return loss


def build_norm_layer(cfg):
    return NormalizationLayer(cfg.MODEL.NORM.SCALE, cfg.MODEL.NORM.LEARNABLE)


def build_loss_func(cfg):
    if cfg.MODEL.LOSS == "bbc":
        if cfg.MODEL.COMP.METHOD == "multi-scale":
            loss_func = MultiScaleBatchBasedClassificationLoss()
        else:
            loss_func = BatchBasedClassificationLoss()
    else:
        raise NotImplementedError
    return loss_func
