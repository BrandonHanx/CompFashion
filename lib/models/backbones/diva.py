"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import pretrainedmodels as ptm
import torch
import torch.nn as nn


class DIVA(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = ptm.__dict__["resnet50"](num_classes=1000, pretrained="imagenet")

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.feature_dim = self.model.last_linear.in_features
        out_dict = nn.ModuleDict()
        for mode in ["discriminative", "selfsimilarity", "shared", "intra"]:
            out_dict[mode] = torch.nn.Linear(self.feature_dim, 512)

        self.model.last_linear = out_dict
        self.layer_blocks = nn.ModuleList(
            [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        )
        self.out_channels = 2048

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        out_dict = {}
        for key, linear_map in self.model.last_linear.items():
            out_dict[key] = linear_map(x)

        return out_dict


def build_diva(cfg):
    model = DIVA()
    model.load_state_dict(torch.load("pretrained/diva_512.pth.tar"))
    return model
