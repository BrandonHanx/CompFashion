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


class BatchBasedSemiHardMarginLoss(nn.Module):
    def __init__(self, margin=0.2, bidirectional=True):
        super().__init__()
        self.margin = margin
        self.bidirectional = bidirectional

    def semihard_negative_miner(self, similarity):
        batch_size = similarity.size(0)
        neg_samples = np.eye(batch_size).bool().logical_not()
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

    def triplet_loss(self, similarity):
        ap_distances = torch.diag(similarity, diagonal=0)
        negative_idxs = self.semihard_negative_miner(similarity.detach().cpu().numpy())
        an_distances = similarity[:, negative_idxs]
        return F.relu(an_distances - ap_distances + self.margin).mean()

    def forward(self, ref_features, tar_features):
        similarity = ref_features.mm(tar_features.transpose(0, 1))
        loss = self.triplet_loss(similarity)
        if self.bidirectional:
            loss = loss + self.triplet_loss(similarity.t())
        return loss


def build_loss_func(loss_type):
    if loss_type == "bbc":
        return BatchBasedClassificationLoss()
    if loss_type == "bbshm":
        return BatchBasedSemiHardMarginLoss()

    raise NotImplementedError
