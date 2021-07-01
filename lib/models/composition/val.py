import torch
import torch.nn as nn


class VAL(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        self.att_ch_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.att_sp_conv = nn.Conv2d(1, 1, 3, padding=1)

        # the number of heads is tunable
        self.mh_att = nn.MultiheadAttention(embed_dim, num_heads=2, bias=False)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 1)

        # weight parameter
        self.a = nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(self, img_feat, text_feat):
        # img_feat : B x H x W x D
        # text_feat : B x D
        img_feat = img_feat.permute(0, 3, 1, 2)
        B, D, H, W = img_feat.size()
        text_feat = text_feat.view(B, D, 1, 1).expand(-1, -1, H, W)  # B x D x H x W
        v1_feat = torch.cat([img_feat, text_feat], 1)  # B x 2D x H x W
        v1_feat = self.conv1(v1_feat)  # B x D x H x W

        gate_sqz = v1_feat.mean((2, 3), keepdim=True)  # B x D x 1 x 1
        att_ch = self.att_ch_conv(gate_sqz)  # B x D x 1 x 1

        gate_sqz = v1_feat.mean(1, keepdim=True)  # B x 1 x H x W
        att_sp = self.att_sp_conv(gate_sqz)  # B x 1 x H x W

        joint_att = torch.sigmoid(att_ch) * torch.sigmoid(att_sp)  # B x D x H x W

        v1_feat = v1_feat.view(B, D, H * W).permute(2, 0, 1)  # H*W x B x D
        self_att, _ = self.mh_att(v1_feat, v1_feat, v1_feat)  # H*W x B x D
        self_att = self_att.view(H, W, B, D).permute(2, 3, 0, 1)  # B x D x H x W
        self_att = self.conv2(self_att)  # B x D x H x W

        composite_features = self.a[0] * joint_att * img_feat + self.a[1] * self_att
        composite_features = composite_features.mean((2, 3))  # B x D
        return composite_features
