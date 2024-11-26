from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from .yolo_pafpn import YOLOPAFPN
from .faster_rcnnfpn import FPN

from ...yolox.models.deoe_head import DEOEHead


def build_head(head_cfg: DictConfig, in_channels: Tuple[int, ...], strides: Tuple[int, ...]):
    head_cfg_dict = OmegaConf.to_container(head_cfg, resolve=True, throw_on_missing=True)
    head_cfg_dict.pop('name')
    head_cfg_dict.pop('version', None)
    head_cfg_dict.update({"in_channels": in_channels})
    head_cfg_dict.update({"strides": strides})
    compile_cfg = head_cfg_dict.pop('compile', None)
    head_cfg_dict.update({"compile_cfg": compile_cfg})
    if head_cfg.name == 'DEOE':
        return DEOEHead(**head_cfg_dict)


def build_yolox_fpn(fpn_cfg: DictConfig, in_channels: Tuple[int, ...]):
    fpn_cfg_dict = OmegaConf.to_container(fpn_cfg, resolve=True, throw_on_missing=True)
    fpn_name = fpn_cfg_dict.pop('name')
    fpn_cfg_dict.update({"in_channels": in_channels})
    if fpn_name in {'PAFPN', 'pafpn'}:
        compile_cfg = fpn_cfg_dict.pop('compile', None)
        fpn_cfg_dict.update({"compile_cfg": compile_cfg})
        return YOLOPAFPN(**fpn_cfg_dict)
    raise NotImplementedError

def build_two_stage_fpn():
    return FPN()
