from torch import nn

from .backbones.bert import build_bert
from .backbones.bigru import build_bigru
from .backbones.resnet import build_resnet
from .composition import build_composition
from .losses import build_loss_evaluator


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.visual_model = None
        self.textual_model = None

        if cfg.MODEL.VISUAL_MODEL in ["resnet34", "resnet50", "resnet101"]:
            self.visual_model = build_resnet(cfg)

        if cfg.MODEL.TEXTUAL_MODEL == "bigru":
            self.textual_model = build_bigru(cfg)
        else:
            self.textual_model = build_bert(cfg)

        self.composition_model = build_composition(
            cfg, self.visual_model.out_channels, self.textual_model.out_channels
        )

        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, batch_data):
        source_feature = self.visual_model(batch_data["source_images"])
        target_feature = self.visual_model(batch_data["target_images"])
        text_feature = self.textual_model(
            batch_data["text"], batch_data["text_lengths"]
        )

        comp_feature = self.composition_model(source_feature, text_feature)

        if self.training:
            losses = self.loss_evaluator(source_feature, target_feature)
            return losses

        return comp_feature


def build_model(cfg):
    return Model(cfg)
