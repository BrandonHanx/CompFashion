import torch
import torch.nn as nn


class MMT(nn.Module):
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
        self.mmt = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        self.v_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.t_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.v_seg_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.t_seg_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.word_seq_length = 30
        self.patch_seq_length = 7 * 7
        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1, self.word_seq_length + self.patch_seq_length + 2, self.embed_dim
            )
        )

    def get_key_padding_mask(self, text_length):
        batch_size = text_length.shape[0]
        mask = torch.zeros(
            batch_size, self.patch_seq_length + self.word_seq_length + 2
        ).to(text_length.device)
        for i in range(batch_size):
            mask[i, text_length[i] + self.patch_seq_length + 2 :] = 1
        return mask.bool()

    def forward(self, patch_seq, word_seq, text_length):
        batch_size = patch_seq.shape[0]
        v_cls_tokens = self.v_cls_token.expand(batch_size, -1, -1)
        t_cls_tokens = self.t_cls_token.expand(batch_size, -1, -1)

        patch_seq = patch_seq + self.v_seg_token

        word_seq = word_seq + self.t_seg_token
        key_padding_mask = self.get_key_padding_mask(text_length)

        seq = torch.cat((v_cls_tokens, patch_seq, t_cls_tokens, word_seq), dim=1)
        seq = seq + self.pos_token
        seq = self.mmt(
            seq.transpose(0, 1), src_key_padding_mask=key_padding_mask
        )  # b x 512 x (49 + 30 + 2)
        return seq[:, :, 1 : 1 + self.patch_seq_length].mean(-1)


def build_mmt(cfg):
    return MMT(cfg.MODEL.COMP.EMBED_DIM, 8, 2048, 0.2, 4)
