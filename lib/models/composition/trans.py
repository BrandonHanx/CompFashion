import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_head,
        ff_dim,
        dropout,
        num_layers,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        self.trans = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, 50, embed_dim))

    def forward(self, patch_seq, word_feat=None):
        cls_token = patch_seq.mean(dim=0, keepdim=True)
        if word_feat is not None:
            patch_seq = patch_seq + word_feat.unsqueeze(1)
        seq = torch.cat((cls_token, patch_seq), dim=1)
        seq = seq + self.pos_embedding
        seq = self.trans(seq.transpose(0, 1))  # 50 x b x 512

        return seq[0]


def build_trans(cfg):
    return Transformer(cfg.MODEL.COMP.EMBED_DIM, 8, 2048, 0.2, 4)
