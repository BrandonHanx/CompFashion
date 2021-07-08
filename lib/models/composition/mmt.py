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

        self.v_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.t_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.zeros(1, 50, embed_dim))

    # def get_key_padding_mask(self, text_length):
    #     batch_size = text_length.shape[0]
    #     mask = torch.zeros(
    #         batch_size, self.patch_seq_length + torch.max(text_length) + 2
    #     ).to(text_length.device)
    #     for i in range(batch_size):
    #         mask[i, text_length[i] + self.patch_seq_length + 2 :] = 1
    #     return mask.bool()

    def forward(self, patch_seq, word_seq):
        patch_seq = patch_seq + self.v_seg_token
        word_seq = word_seq.unsqueeze(1) + self.t_seg_token

        seq = torch.cat((patch_seq, word_seq), dim=1)
        seq = seq + self.pos_embedding
        seq = self.mmt(seq.transpose(0, 1))  # 50 x b x 512

        return seq[:-1].transpose(0, 1)


def build_mmt(cfg):
    return MMT(cfg.MODEL.COMP.EMBED_DIM, 8, 2048, 0.2, 4)
