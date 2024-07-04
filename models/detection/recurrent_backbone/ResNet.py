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
    
from models.layers.ResNet.backbone import (
    Bottleneck
)
from models.layers.rnn import DWSConvLSTM2d

from .base import BaseDetector
from models.layers.maxvit.maxvit import (
    nhwC_2_nChw)

class Detector(BaseDetector):
    def __init__(self, mdl_config: DictConfig,):
        super().__init__()

        ###### Config ######
        self.inplanes = 32
        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)
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
        self.stage_dims = [64, 128, 256, 512]

        self.stages = nn.ModuleList()
        self.strides = [] 

        layers = [3, 4, 23, 3]

        planes = [32, 64, 128, 256]

        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=7, stride=2, padding=3,
                 bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

     
        for stage_idx, (plane, layer) in \
                enumerate(zip(planes, layers)):
            
            spatial_downsample_factor = 4 if stage_idx == 0 else 2

            if stage_idx == 0:
                stage = ResNet_lstm(self.inplanes, plane, layer, stride = 1, lstm_cfg = mdl_config.stage.lstm, stage_dim = self.stage_dims[stage_idx])
            else:
                stage = ResNet_lstm(self.inplanes, plane, layer, stride = 2, lstm_cfg = mdl_config.stage.lstm, stage_dim = self.stage_dims[stage_idx])
            
            self.inplanes = self.inplanes * 2
            stride = stride * spatial_downsample_factor

            self.strides.append(stride)
            self.stages.append(stage)
            
        self.stage_dims = self.stage_dims
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
    
    def forward(self, x: th.Tensor,
                prev_states: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None) \
            -> Tuple[BackboneFeatures]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
    
        output: Dict[int, FeatureMap] = {}
        states: LstmStates = list()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for stage_idx, stage in enumerate(self.stages):
            x, state = stage(x, prev_states[stage_idx])
            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x

        return output, states
    

class ResNet_lstm(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self, init_planes, plane, 
                 layer, stride, 
                 lstm_cfg = None, stage_dim = 0, 
                 memory_type: str = 'lstm',):

        super().__init__()
        self.memory_type = memory_type

        self.extractor = self._make_layer(Bottleneck, init_planes, plane, layer, stride)
  
        if memory_type == 'lstm':
            self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                    dws_conv=lstm_cfg.dws_conv,
                                    dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                    dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                    cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
            
    def _make_layer(self, block, init_planes, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or init_planes != planes * block.expansion:
            downsample = nn.Sequential(
            nn.Conv2d(init_planes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(init_planes, planes, stride, downsample))
        init_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(init_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None) \
            -> Tuple[FeatureMap, LstmState]:

     
        x = self.extractor(x)  # N C H W -> N H W C
        # x = nhwC_2_nChw(x)  # N H W C -> N C H W
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple




