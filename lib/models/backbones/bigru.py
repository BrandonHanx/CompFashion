import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(
        self,
        hidden_dim,
        vocab_size,
        embed_size,
        num_layers,
        drop_out,
    ):
        super().__init__()

        if vocab_size == embed_size:
            self.embed = None
        else:
            self.embed = nn.Linear(vocab_size, embed_size)

        self.gru = nn.GRU(
            embed_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=drop_out,
            bidirectional=True,
            bias=False,
        )
        self.out_channels = hidden_dim * 2
        self._init_weight()

    def forward(self, text, text_length):
        if self.embed is not None:
            text = self.embed(text)

        gru_out = self.gru_out(text, text_length)
        gru_out, _ = torch.max(gru_out, dim=1)
        return gru_out

    def gru_out(self, embed, text_length):
        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        gru_sort_out, _ = self.gru(pack)
        gru_sort_out = nn.utils.rnn.pad_packed_sequence(gru_sort_out, batch_first=True)
        gru_sort_out = gru_sort_out[0]

        gru_out = gru_sort_out.index_select(0, idx_unsort)
        return gru_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)


class BiGRUwithEmbeddingLayer(BiGRU):
    def __init__(
        self,
        hidden_dim,
        vocab_size,
        embed_size,
        num_layers,
        drop_out,
    ):
        super().__init__(
            hidden_dim,
            vocab_size,
            embed_size,
            num_layers,
            drop_out,
        )
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

    def forward(self, text, text_length):
        text = self.embed(text)
        gru_out = self.gru_out(text, text_length)
        gru_out, _ = torch.max(gru_out, dim=1)
        return gru_out


def build_bigru(cfg):
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = cfg.MODEL.GRU.DROPOUT

    if cfg.MODEL.VOCAB == "init":
        model = BiGRUwithEmbeddingLayer(
            hidden_dim,
            vocab_size,
            embed_size,
            num_layer,
            drop_out,
        )
    else:
        model = BiGRU(
            hidden_dim,
            vocab_size,
            embed_size,
            num_layer,
            drop_out,
        )

    if cfg.MODEL.GRU.FREEZE:
        for m in [model.embed, model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
