from typing import Dict, Optional, Tuple
import ipdb
import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from data.utils.types import FeatureMap, BackboneFeatures
from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None
from models.layers.FWLR.BFM import (
    Temporal_Active_Focus_connect,
    BaseConv,
    ResLayer,
    SPPBottleneck
    )
from models.layers.rnn import DWSConvLSTM2d

from models.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)

from .base import BaseDetector

class Detector(BaseDetector):
    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        ###### Config ######
        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)
        FRLW_channels = mdl_config.FRLW_channels
        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        ##################################

        input_dim = in_channels
        stride = 1
        self.stage_dims = FRLW_channels

        self.stages = nn.ModuleList()
        self.strides = []

        self.stem = Temporal_Active_Focus_connect(input_dim, self.stage_dims[0],ksize=3)

        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = 4 if stage_idx == 0 else 2

            if stage_idx<=2:
                stage = nn.Sequential(
                *self.make_group_layer(self.stage_dims[stage_idx], self.stage_dims[stage_idx+1], num_blocks, stride=2))
            else:
                stage = nn.Sequential(
                *self.make_group_layer(self.stage_dims[stage_idx], self.stage_dims[stage_idx+1], num_blocks, stride=2),
                *self.make_spp_block([self.stage_dims[stage_idx], self.stage_dims[stage_idx]], self.stage_dims[stage_idx],),
                )


            stride = stride * spatial_downsample_factor
            self.strides.append(stride)
            self.stages.append(stage)
            
        self.stage_dims = self.stage_dims[1:]
        self.num_stages = num_stages

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)
    
    def make_group_layer(self, in_channels, out_channels, num_blocks, stride, act = "silu"):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, out_channels, ksize=3, stride=stride,act=act),
            *[(ResLayer(out_channels,act=act)) for _ in range(num_blocks)],
        ]
    
    def make_spp_block(self, filters_list, in_filters, act="silu"):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act=act),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation=act,
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act=act),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act=act),
            ]
        )
        return m

    def forward(self, x: th.Tensor) \
            -> Tuple[BackboneFeatures]:
        
        output: Dict[int, FeatureMap] = {}
        x = self.stem(x)
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            ipdb.set_trace()
            stage_number = stage_idx + 1
            output[stage_number] = x
        return output




