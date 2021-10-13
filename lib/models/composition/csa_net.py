import torch
import torch.nn as nn


class CSANet(nn.Module):
    def __init__(self, n_conditions, text_channel, img_channel):
        super().__init__()
        self.num_conditions = n_conditions
        self.weight_classifier = nn.Sequential(
            nn.Linear(text_channel, n_conditions),
            nn.Softmax(),
        )
        self.masks = torch.nn.Embedding(n_conditions, img_channel)

    def forward(self, x, c):
        weight = self.weight_classifier(c)  # B x 5
        x = x.unsqueeze(1).expand(-1, 5, -1)  # B x 5 x D
        x = x * self.masks.weight.unsqueeze(0)  # B x 5 x D
        x = x * weight.unsqueeze(-1)  # B x 5 x D
        return x.mean(dim=1)


def build_csanet(text_channel, img_channel):
    return CSANet(5, text_channel, img_channel)
