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
        self.start_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.zeros(1, 257, embed_dim))
        self.tgt_mask = torch.tril(torch.ones(257, 257)).cuda()

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1024, bias=False)

    def forward(self, patch_indices, word_seq, tgt_indices):
        patch_seq = self.v_embeddings(patch_indices)
        tgt_seq = self.v_embeddings(tgt_indices)

        patch_seq = patch_seq + self.v_seg_token
        word_seq = word_seq.unsqueeze(1) + self.t_seg_token

        seq = torch.cat((word_seq, patch_seq), dim=1)
        seq = seq + self.pos_embedding

        start_tokens = self.start_token.expand(patch_indices.shape[0], -1, -1)
        tgt_seq = torch.cat((start_tokens, tgt_seq), dim=1)
        tgt_seq = tgt_seq + self.pos_embedding
        seq = self.transformer(seq, tgt_seq, tgt_mask=self.tgt_mask)  # b x 257 x 512

        logits = self.head(self.ln_f(seq[:, 1:]))

        return logits

    @torch.no_grad()
    def sample(self, patch_indices, word_seq):
        patch_seq = self.v_embeddings(patch_indices)
        patch_seq = patch_seq + self.v_seg_token
        word_seq = word_seq.unsqueeze(1) + self.t_seg_token

        seq = torch.cat((word_seq, patch_seq), dim=1)
        seq = seq + self.pos_embedding
        start_tokens = self.start_token.expand(patch_indices.shape[0], -1, -1)

        memory = self.transformer.encoder(seq)
        tgt_indices = None

        for i in range(0, 256):
            if tgt_indices is None:
                tgt_seq = start_tokens
            else:
                tgt_seq = torch.cat(
                    (start_tokens, self.v_embeddings(tgt_indices)), dim=1
                )
            tgt_seq = tgt_seq + self.pos_embedding[:, : i + 1]
            pred_token = self.transformer.decoder(tgt_seq, memory)
            pred_token = pred_token[:, -1]
            pred_logits = self.head(self.ln_f(pred_token)).squeeze()
            pred_indice = torch.argmax(pred_logits, dim=-1, keepdim=True)
            if tgt_indices is None:
                tgt_indices = pred_indice
            else:
                tgt_indices = torch.cat((tgt_indices, pred_indice), dim=-1)

        return tgt_indices


def build_transdec(cfg):
    return TransDec(cfg.MODEL.COMP.EMBED_DIM)
