import torch
import torch.nn as nn


class TransEnc(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_head,
        ff_dim,
        dropout,
        num_layers,
    ):
        super().__init__()

        self.v_embeddings = nn.Embedding(1024, embed_dim)
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

        self.v_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.t_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.zeros(1, 257, embed_dim))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1024, bias=False)

    def forward(self, patch_indices, word_seq, tgt_indices):
        patch_seq = self.v_embeddings(patch_indices)

        patch_seq = patch_seq + self.v_seg_token
        word_seq = word_seq.unsqueeze(1) + self.t_seg_token

        seq = torch.cat((word_seq, patch_seq), dim=1)
        seq = seq + self.pos_embedding

        seq = self.trans(seq)  # b x 257 x 512

        logits = self.head(self.ln_f(seq[:, 1:]))

        return logits

    @torch.no_grad()
    def sample(self, patch_indices, word_seq):
        return self.forward(patch_indices, word_seq, None)


def build_transenc(cfg):
    return TransEnc(cfg.MODEL.COMP.EMBED_DIM, 8, 2048, 0.2, 4)
