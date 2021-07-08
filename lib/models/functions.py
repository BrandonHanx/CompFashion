import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_norm_layer", "build_loss_func", "build_attn_pool"]


class NormalizationLayer(nn.Module):
    """Class for normalization layer."""

    def __init__(self, normalize_scale=4.0, learn_scale=True):
        super().__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x, dim=1):
        features = self.norm_s * x / torch.norm(x, dim=dim, keepdim=True).expand_as(x)
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


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim,
        embed_dim,
        num_heads,
        output_dim=None,
        patch_size=1,
    ):
        super().__init__()
        self.spacial_dim = spacial_dim
        self.proj_conv = None
        if patch_size > 1:
            self.proj_conv = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
        self.positional_embedding = nn.Parameter(
            torch.randn(
                (spacial_dim[0] // patch_size) * (spacial_dim[1] // patch_size) + 1,
                embed_dim,
            )
            / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x[0]


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


def build_attn_pool(cfg):
    return AttentionPool2d(
        spacial_dim=(7, 7), embed_dim=cfg.MODEL.COMP.EMBED_DIM, num_heads=32
    )
