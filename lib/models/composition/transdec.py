import torch
import torch.nn as nn


class TransDec(nn.Module):
    def __init__(
        self,
        embed_dim,
    ):
        super().__init__()

        self.v_embeddings = nn.Embedding(1024, embed_dim)
        self.transformer = nn.Transformer(d_model=embed_dim, batch_first=True)

        self.v_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.t_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.zeros(1, 257, embed_dim))
        self.tgt_mask = torch.tril(torch.ones(256, 256))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1024, bias=False)

    def forward(self, patch_indices, word_seq, tgt_seq):
        patch_seq = self.v_embeddings(patch_indices)
        patch_seq = patch_seq + self.v_seg_token
        word_seq = word_seq.unsqueeze(1) + self.t_seg_token

        seq = torch.cat((patch_seq, word_seq), dim=1)
        seq = seq + self.pos_embedding
        seq = self.transformer(seq, tgt_seq, tgt_mask=self.tgt_mask)  # b x 256 x 512

        logits = self.head(self.ln_f(seq))

        return logits


def build_transdec(cfg):
    return TransDec(cfg.MODEL.COMP.EMBED_DIM)
