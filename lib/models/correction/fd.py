import torch
import torch.nn as nn


class FusDiff(nn.Module):
    def __init__(self, embed_dim, img_channel):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim + img_channel, embed_dim), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(embed_dim + img_channel, embed_dim), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + img_channel * 2, embed_dim), nn.ReLU()
        )

    def forward(self, x_before, x_after):

        x_before_ = self.fc1(torch.cat((x_before * x_after, x_before), -1))  # B x D
        x_after_ = self.fc2(torch.cat((x_before * x_after, x_after), -1))  # B x D
        x_diff = x_after_ - x_before_

        x = torch.cat((x_before, x_diff, x_after), -1)  # B x 3*D
        x = self.fc(x)

        return x


def build_fd(cfg, img_channel):
    return FusDiff(cfg.MODEL.COMP.EMBED_DIM, img_channel)
