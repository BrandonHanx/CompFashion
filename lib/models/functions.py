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


class KLDiv(nn.Module):
    @staticmethod
    def forward(pred, label):
        return {
            "kl_div": F.kl_div(
                F.log_softmax(pred, -1), F.softmax(label, -1), reduction="batchmean"
            )
        }


def build_norm_layer(cfg):
    return NormalizationLayer(cfg.MODEL.NORM.SCALE, cfg.MODEL.NORM.LEARNABLE)


def build_loss_func(cfg):
    def build_loss_func_(loss_type):
        if loss_type == "bbc":
            loss_func = BatchBasedClassificationLoss()
        elif loss_type == "kl_div":
            loss_func = KLDiv()
        else:
            raise NotImplementedError
        return loss_func

    loss_types = cfg.MODEL.LOSS.split("+")
    if len(loss_types) == 1:
        return build_loss_func_(loss_types[0])

    loss_funcs = []
    for loss_type in loss_types:
        loss_funcs.append(build_loss_func_(loss_type))
    return nn.ModuleList(loss_funcs)
