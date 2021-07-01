import torch
import torch.nn as nn

from .attention import AttentionModule
from .utils import EqualLinear, calculate_mean_std


class GlobalStyleTransformer(nn.Module):
    def __init__(self, feature_size, text_feature_size, *args, **kwargs):
        super().__init__()
        self.global_transform = EqualLinear(text_feature_size, feature_size * 2)
        self.gate = EqualLinear(text_feature_size, feature_size * 2)
        self.sigmoid = nn.Sigmoid()

        self.init_style_weights(feature_size)

    def forward(self, normed_x, t, *args, **kwargs):
        x_mu, x_std = calculate_mean_std(kwargs["x"])
        gate = self.sigmoid(self.gate(t)).unsqueeze(-1).unsqueeze(-1)
        std_gate, mu_gate = gate.chunk(2, 1)

        global_style = self.global_transform(t).unsqueeze(2).unsqueeze(3)
        gamma, beta = global_style.chunk(2, 1)

        gamma = std_gate * x_std + gamma
        beta = mu_gate * x_mu + beta
        out = gamma * normed_x + beta
        return out

    def init_style_weights(self, feature_size):
        self.global_transform.linear.bias.data[:feature_size] = 1
        self.global_transform.linear.bias.data[feature_size:] = 0


class DisentangledTransformer(nn.Module):
    def __init__(
        self,
        feature_size,
        text_feature_size,
        num_heads,
        global_styler=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        self.att_module = AttentionModule(
            feature_size, text_feature_size, num_heads, *args, **kwargs
        )
        self.att_module2 = AttentionModule(
            feature_size, text_feature_size, num_heads, *args, **kwargs
        )
        self.global_styler = global_styler

        self.weights = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.instance_norm = nn.InstanceNorm2d(feature_size)

    def forward(self, x, t, *args, **kwargs):
        normed_x = self.instance_norm(x)
        att_out, _ = self.att_module(normed_x, t, return_map=True)
        out = normed_x + self.weights[0] * att_out

        att_out2, _ = self.att_module2(out, t, return_map=True)
        out = out + self.weights[1] * att_out2

        out = self.global_styler(out, t, x=x)
        return out


def build_cosmo(cfg):
    embed_dim = cfg.MODEL.COMP.EMBED_DIM
    global_styler = GlobalStyleTransformer(2048, embed_dim)
    cosmo = DisentangledTransformer(
        2048, embed_dim, num_heads=8, global_styler=global_styler
    )
    return cosmo
