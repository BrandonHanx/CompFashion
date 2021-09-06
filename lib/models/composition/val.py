import torch
import torch.nn as nn


class VAL(nn.Module):
    def __init__(self, text_channel, img_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(text_channel + img_channel, img_channel, 1)
        self.att_ch_conv = nn.Conv2d(img_channel, img_channel, 1)
        self.att_sp_conv = nn.Conv2d(1, 1, 3, padding=1)

        # the number of heads is tunable
        self.mh_att = nn.MultiheadAttention(img_channel, num_heads=2, bias=False)
        self.conv2 = nn.Conv2d(img_channel, img_channel, 1)

        # weight parameter
        self.a = nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(self, img_feat, text_feat):
        # img_feat : B x H x W x C
        # text_feat : B x D
        B, C, H, W = img_feat.size()
        _, D = text_feat.size()

        text_feat = text_feat.view(B, D, 1, 1).expand(-1, -1, H, W)  # B x D x H x W
        v1_feat = torch.cat([img_feat, text_feat], 1)  # B x (D + C) x H x W
        v1_feat = self.conv1(v1_feat)  # B x C x H x W

        gate_sqz = v1_feat.mean((2, 3), keepdim=True)  # B x C x 1 x 1
        att_ch = self.att_ch_conv(gate_sqz)  # B x C x 1 x 1

        gate_sqz = v1_feat.mean(1, keepdim=True)  # B x 1 x H x W
        att_sp = self.att_sp_conv(gate_sqz)  # B x 1 x H x W

        joint_att = torch.sigmoid(att_ch) * torch.sigmoid(att_sp)  # B x C x H x W

        v1_feat = v1_feat.view(B, C, H * W).permute(2, 0, 1)  # H*W x B x C
        self_att, _ = self.mh_att(v1_feat, v1_feat, v1_feat)  # H*W x B x C
        self_att = self_att.view(H, W, B, C).permute(2, 3, 0, 1)  # B x C x H x W
        self_att = self.conv2(self_att)  # B x C x H x W

        composite_features = self.a[0] * joint_att * img_feat + self.a[1] * self_att
        composite_features = composite_features.mean((2, 3))  # B x C
        return composite_features


def build_val(text_channel, img_channel):
    return VAL(text_channel, img_channel)
