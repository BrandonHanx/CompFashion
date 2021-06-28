import torch
import torch.nn as nn


class TIRG(nn.Module):
    """The TIGR model.
    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, embed_dim):
        super().__init__()

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim),
        )
        self.res_info_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, img_features, text_features):
        x = torch.cat((img_features, text_features), dim=1)
        f1 = self.gated_feature_composer(x)
        f2 = self.res_info_composer(x)
        f = torch.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
        return f


def build_tirg(cfg):
    return TIRG(cfg.MODEL.COMP_MODEL.EMBED_DIM)
