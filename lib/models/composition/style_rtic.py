import torch
import torch.nn as nn

from .cosmo import calculate_mean_std
from .rtic import ErrorEncodingBlock, GatingBlock


class ConvFushionBlock(nn.Module):
    def __init__(self, text_dim, img_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(text_dim + img_dim)
        self.activation = nn.LeakyReLU()
        self.conv = nn.Linear(text_dim + img_dim, img_dim)

    def forward(self, img_feat, text_feat):
        B, _, H, W = img_feat.size()
        _, D = text_feat.size()

        text_feat = text_feat.view(B, D, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([img_feat, text_feat], dim=1)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class ConvErrorEncodingBlock(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        half = int(img_dim / 2)
        self.sub_block_1 = nn.Sequential(
            nn.Conv2d(img_dim, half),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(),
        )
        self.sub_block_2 = nn.Sequential(
            nn.Conv2d(half, half),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(),
        )
        self.conv_3 = nn.Conv2d(half, img_dim)

    def forward(self, x):
        residual = x
        x = self.sub_block_1(x)
        x = self.sub_block_2(x)
        x = self.conv_3(x)
        return x + residual


class ConvGatingBlock(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.sub_block_1 = nn.Sequential(
            nn.Conv2d(img_dim, img_dim),
            nn.BatchNorm2d(img_dim),
            nn.LeakyReLU(),
        )
        self.sub_block_2 = nn.Sequential(
            nn.Conv2d(img_dim, img_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.sub_block_1(x)
        x = self.sub_block_2(x)
        return x


class StyleRTIC(nn.Module):
    def __init__(self, text_dim, img_dim, n=4):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(img_dim)
        self.fusion_block = ConvFushionBlock(text_dim, img_dim)

        self.content_encoding = nn.ModuleList([ConvErrorEncodingBlock(img_dim)] * n)
        self.content_gating = ConvGatingBlock(img_dim)

        self.scale_encoding = nn.ModuleList([ErrorEncodingBlock(img_dim)] * n)
        self.scale_gating = GatingBlock(img_dim)

        self.bias_encoding = nn.ModuleList([ErrorEncodingBlock(img_dim)] * n)
        self.bias_gating = GatingBlock(img_dim)

    def forward(self, img_feats, text_feats):
        img_mu, img_std = calculate_mean_std(img_feats)  # B x C
        norm_img_feats = self.instance_norm(img_feats)  # B x C x H x W
        fused_featmaps = self.fusion_block(img_feats, text_feats)  # B x C x H x W
        fused_feats = fused_featmaps.mean((2, 3))

        content_gating = self.content_gating(fused_featmaps)
        content_feats = fused_featmaps
        for block in self.content_encoding:
            content_feats = block(content_feats)
        content_feats = norm_img_feats * content_gating + content_feats * (
            1 - content_gating
        )

        scale_gating = self.scale_gating(fused_feats)
        scale_feats = fused_feats
        for block in self.scale_encoding:
            scale_feats = block(scale_feats)
        scale_feats = img_std * scale_gating + scale_feats * (1 - scale_gating)

        bias_gating = self.bias_gating(fused_feats)
        bias_feats = fused_feats
        for block in self.bias_encoding:
            bias_feats = block(bias_feats)
        bias_feats = img_mu * bias_gating + bias_feats * (1 - bias_gating)

        return scale_feats * content_feats.mean((2, 3)) + bias_feats


def build_style_rtic(cfg, img_channel):
    return StyleRTIC(cfg.MODEL.COMP.EMBED_DIM, img_channel, n=4)
