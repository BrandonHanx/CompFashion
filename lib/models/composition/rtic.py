import torch
import torch.nn as nn


class FushionBlock(nn.Module):
    def __init__(self, text_dim, img_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(text_dim + img_dim)
        self.activation = nn.LeakyReLU()
        self.fc = nn.Linear(text_dim + img_dim, img_dim)

    def forward(self, img_feats, text_feats):
        x = torch.cat([img_feats, text_feats], dim=-1)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc(x)


class ErrorEncodingBlock(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.sub_block_1 = nn.Sequential(
            nn.Linear(img_dim, img_dim / 2),
            nn.BatchNorm1d(img_dim / 2),
            nn.LeakyReLU(),
        )
        self.sub_block_2 = nn.Sequential(
            nn.Linear(img_dim / 2, img_dim / 2),
            nn.BatchNorm1d(img_dim / 2),
            nn.LeakyReLU(),
        )
        self.fc_3 = nn.Linear(img_dim / 2, img_dim)

    def forward(self, x):
        residual = x
        x = self.sub_block_1(x)
        x = self.sub_block_2(x)
        x = self.fc_3(x)
        return x + residual


class GatingBlock(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.sub_block_1 = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.BatchNorm1d(img_dim),
            nn.LeakyReLU(),
        )
        self.sub_block_2 = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.sub_block_1(x)
        x = self.sub_block_2(x)
        return x


class RTIC(nn.Module):
    def __init__(self, text_dim, img_dim, n=4):
        self.fusion_block = FushionBlock(text_dim, img_dim)
        self.error_encoding_blocks = nn.ModuleList([ErrorEncodingBlock(img_dim)] * n)
        self.gating_block = GatingBlock(img_dim)

    def forward(self, img_feats, text_feats):
        x = self.fusion_block(img_feats, text_feats)
        s = self.gating_block(x)
        for error_encoding_block in self.error_encoding_blocks:
            x = error_encoding_block(x)
        x = img_feats * s + x * (1 - s)
        return x


def build_rtic(cfg, img_channel):
    return RTIC(cfg.MODEL.COMP.EMBED_DIM, img_channel, n=4)
