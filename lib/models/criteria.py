import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_loss_func"]


class BatchBasedClassificationLoss(nn.Module):
    @staticmethod
    def forward(ref_features, tar_features, bmm=False):
        batch_size = ref_features.size(0)
        device = ref_features.device

        if bmm:
            pred = torch.bmm(ref_features, tar_features.unsqueeze(-1)).squeeze()
        else:
            pred = ref_features.mm(tar_features.transpose(0, 1))

        labels = torch.arange(0, batch_size).long().to(device)
        loss = F.cross_entropy(pred, labels)
        return loss


class DistanceAdaptedBatchBasedClassificationLoss(nn.Module):
    def __init__(self, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lower_cutoff = 0.5
        self.upper_cutoff = 1.4

    def get_distance_weights(self, anchor_features, negative_features):
        dim = anchor_features.shape[-1]
        anchor_features = F.normalize(anchor_features, dim=-1)
        negative_features = F.normalize(negative_features, dim=-1)
        anchor_negative_euclid = torch.cdist(
            anchor_features, negative_features
        ).squeeze()  # B x B-1
        anchor_negative_euclid = anchor_negative_euclid.clamp(min=self.lower_cutoff)

        distance_weighting = (2.0 - dim) * torch.log(anchor_negative_euclid) - (
            dim - 3 / 2
        ) * torch.log(1.0 - 0.25 * (anchor_negative_euclid.pow(2)))
        max_per_row, _ = torch.max(distance_weighting, dim=1, keepdim=True)
        distance_weighting = torch.exp(distance_weighting - max_per_row)
        distance_weighting[anchor_negative_euclid > self.upper_cutoff] = 0
        distance_weighting = distance_weighting.clamp(min=1e-45)
        distance_weighting = distance_weighting / torch.sum(
            distance_weighting, dim=1, keepdim=True
        )
        return distance_weighting

    def get_loss(self, anchor_features, tar_features):
        batch_size = anchor_features.shape[0]
        device = anchor_features.device

        tar_features = tar_features.view(1, batch_size, -1).expand(
            batch_size, -1, -1
        )  # B x B x D
        positive_mask = (
            torch.eye(batch_size).bool().unsqueeze(-1).to(device)
        )  # B x B x 1
        positive_features = tar_features.masked_select(positive_mask).view(
            batch_size, 1, -1
        )  # B x 1 x D
        negative_features = tar_features.masked_select(~positive_mask).view(
            batch_size, batch_size - 1, -1
        )  # B x B-1 x D
        anchor_features = anchor_features.unsqueeze(-2)  # B x 1 x D

        positive_dist = torch.bmm(
            anchor_features, positive_features.transpose(1, 2)
        ).squeeze(
            1
        )  # B x 1
        negative_dist = torch.bmm(
            anchor_features, negative_features.transpose(1, 2)
        ).squeeze(
            1
        )  # B x B-1

        distance_weighting = self.get_distance_weights(
            anchor_features.detach(), negative_features.detach()
        )

        labels = torch.zeros(batch_size).long().to(device)
        pred = torch.cat([positive_dist, negative_dist * distance_weighting], dim=1)
        loss = F.cross_entropy(pred, labels)
        return loss

    def forward(self, ref_features, tar_features):
        loss = self.get_loss(ref_features, tar_features)
        if self.bidirectional:
            loss = loss + self.get_loss(tar_features, ref_features)
        #             loss = loss / 2
        return loss


class BatchBasedMarginLoss(nn.Module):
    def __init__(self, margin=0.8, bidirectional=True):
        super().__init__()
        self.margin = margin
        self.bidirectional = bidirectional

    def negative_miner(self, similarity):
        raise NotImplementedError

    def triplet_loss(self, similarity):
        ap_distances = torch.diag(similarity, diagonal=0)
        negative_idxs = self.negative_miner(similarity.detach().cpu().numpy())
        an_distances = similarity[:, negative_idxs]
        return F.relu(an_distances - ap_distances + self.margin).mean()

    def forward(self, ref_features, tar_features):
        similarity = ref_features.mm(tar_features.transpose(0, 1))
        loss = self.triplet_loss(similarity)
        if self.bidirectional:
            loss = loss + self.triplet_loss(similarity.t())
        return loss


class BatchBasedHardMarginLoss(BatchBasedMarginLoss):
    def negative_miner(self, similarity):
        batch_size = similarity.shape[0]
        pos_samples = np.eye(batch_size).astype(bool)
        similarity[pos_samples] = -1
        negatives = np.argmax(similarity, axis=1)
        return list(negatives)


class BatchBasedSemiHardMarginLoss(BatchBasedMarginLoss):
    def negative_miner(self, similarity):
        batch_size = similarity.shape[0]
        neg_samples = ~np.eye(batch_size).astype(bool)
        negatives = []
        for i in range(batch_size):
            pos_dist = similarity[i, i]
            neg_sample = neg_samples[i]
            neg_mask = np.logical_and(neg_sample, similarity[i] > pos_dist)
            neg_mask = np.logical_and(neg_mask, similarity[i] < self.margin + pos_dist)
            if neg_mask.sum() > 0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg_sample)[0]))
        return negatives


def build_loss_func(cfg):
    loss_type = cfg.MODEL.LOSS
    if loss_type == "bbc":
        return BatchBasedClassificationLoss()
    if loss_type == "bbshm":
        return BatchBasedSemiHardMarginLoss()
    if loss_type == "bbhm":
        return BatchBasedHardMarginLoss()
    if loss_type == "dabbc":
        return DistanceAdaptedBatchBasedClassificationLoss()
    raise NotImplementedError
