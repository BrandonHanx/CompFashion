import torch
import torch.nn as nn


class LSTM(nn.Module):
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

        self.lstm = nn.LSTM(
            embed_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=drop_out,
            bidirectional=False,
            bias=False,
        )
        self.out_channels = hidden_dim
        self._init_weight()

    def forward(self, text, text_length):
        if self.embed is not None:
            text = self.embed(text)

        lstm_out = self.lstm_out(text, text_length)
        lstm_out, _ = torch.max(lstm_out, dim=1)
        return lstm_out

    def lstm_out(self, embed, text_length):
        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        lstm_sort_out, _ = self.lstm(pack)
        lstm_sort_out = nn.utils.rnn.pad_packed_sequence(
            lstm_sort_out, batch_first=True
        )
        lstm_sort_out = lstm_sort_out[0]

        lstm_out = lstm_sort_out.index_select(0, idx_unsort)
        return lstm_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)


class LSTMwithEmbeddingLayer(LSTM):
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
        lstm_out = self.lstm_out(text, text_length)
        lstm_out, _ = torch.max(lstm_out, dim=1)
        return lstm_out


def build_lstm(cfg):
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = cfg.MODEL.GRU.DROPOUT

    if cfg.MODEL.VOCAB == "init":
        model = LSTMwithEmbeddingLayer(
            hidden_dim,
            vocab_size,
            embed_size,
            num_layer,
            drop_out,
        )
    else:
        model = LSTM(
            hidden_dim,
            vocab_size,
            embed_size,
            num_layer,
            drop_out,
        )

    if cfg.MODEL.GRU.FREEZE:
        for m in [model.embed, model.lstm]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
