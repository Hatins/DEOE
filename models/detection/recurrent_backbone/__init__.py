from omegaconf import DictConfig

from .maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .FWLR import Detector as FWLR
from .ResNet import Detector as ResNet
import ipdb


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name

    if name == 'MaxViTRNN':
        return MaxViTRNNDetector(backbone_cfg)
    elif name == 'FRLW':
        return FWLR(backbone_cfg)
    elif name == 'ResNet':
        return ResNet(backbone_cfg)
    else:
        raise NotImplementedError
