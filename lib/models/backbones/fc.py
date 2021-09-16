import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, int(out_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(out_dim / 4), int(out_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(out_dim / 2), out_dim),
        )
        self.out_channels = out_dim

    def forward(self, text, text_length):
        return self.model(text)


def build_fc(cfg):
    return FC(54, 2048)
