from typing import Dict, Optional, Tuple
import ipdb
import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None
from einops import rearrange
from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from models.layers.rnn import DWSConvLSTM2d
from models.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)


from .base import BaseDetector


class RNNDetector(BaseDetector):
    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        ###### Config ######
        in_channels = mdl_config.input_channels

        memory_type = mdl_config.memory_type
        self.memory_type = memory_type 

        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)
        enable_masking = mdl_config.enable_masking

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
        patch_size = mdl_config.stem.patch_size
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.stages = nn.ModuleList()
        self.strides = []
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            stage = RNNDetectorStage(dim_in=input_dim,
                                     stage_dim=stage_dim,
                                     spatial_downsample_factor=spatial_downsample_factor,
                                     num_blocks=num_blocks,
                                     enable_token_masking=enable_masking_in_stage,
                                     T_max_chrono_init=T_max_chrono_init_stage,
                                     stage_cfg=mdl_config.stage,
                                     memory_type = memory_type)
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)
            input_dim = stage_dim
            self.stages.append(stage)

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

    def forward(self, x: th.Tensor, prev_states: Optional[LstmStates] = None, token_mask: Optional[th.Tensor] = None) \
            -> Tuple[BackboneFeatures, LstmStates]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        states: LstmStates = list()
        output: Dict[int, FeatureMap] = {}
        for stage_idx, stage in enumerate(self.stages):
            x, state = stage(x, prev_states[stage_idx], token_mask if stage_idx == 0 else None)
            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x
        return output, states


class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim: int,
                 skip_first_norm: bool,
                 attention_cfg: DictConfig):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x


class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: DictConfig,
                 memory_type: str = 'lstm',
                 time_scale: float = 1.0):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        self.memory_type = memory_type
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)

        if memory_type == 'lstm':
            self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                    dws_conv=lstm_cfg.dws_conv,
                                    dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                    dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                    cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
            
        elif memory_type == 's5':
            from models.layers.s5.s5_model import S5Block
            self.s5_block = S5Block(
            dim=stage_dim, state_dim=stage_dim, bidir=False, bandlimit=0.5
        )

        ###### Mask Token ################
        self.mask_token = nn.Parameter(th.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            th.nn.init.normal_(self.mask_token, std=.02)
        ##################################

    def forward_ssm(self,
        x: th.Tensor,
        states: Optional[LstmState] = None,
        token_mask: Optional[th.Tensor] = None,
        train_step: bool = True,
    ) -> Tuple[FeatureMap, LstmState]:
        sequence_length = x.shape[0]
        batch_size = x.shape[1]
        x = rearrange(x, "L B C H W -> (L B) C H W")  # where B' = (L B) is the new batch size
        x = self.downsample_cf2cl(x)  # B' C H W -> B' H W C

        if token_mask is not None:
            assert self.mask_token is not None, "No mask token present in this stage"
            x[token_mask] = self.mask_token

        for blk in self.att_blocks:
            x = blk(x)

        x = nhwC_2_nChw(x)  # B' H W C -> B' C H W
        new_h, new_w = x.shape[2], x.shape[3]
        x = rearrange(x, "(L B) C H W -> (B H W) L C", L=sequence_length)

        if states is None:
            states = self.s5_block.s5.initial_state(batch_size=batch_size * new_h * new_w).to(x.device)
        else:
            states = rearrange(states, "B C H W -> (B H W) C")

        x, states = self.s5_block(x, states)
        x = rearrange(x, "(B H W) L C -> L B C H W", B=batch_size, H=int(new_h), W=int(new_w))
        states = rearrange(states, "(B H W) C -> B C H W", H=new_h, W=new_w)

        return x, states


    def forward_rnn(self, x: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None) \
            -> Tuple[FeatureMap, LstmState]:
        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token
        for blk in self.att_blocks:
            x = blk(x)
        x = nhwC_2_nChw(x)  # N H W C -> N C H W
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple
    
    def forward(self, x: th.Tensor,
            h_and_c_previous: Optional[LstmState] = None,
            token_mask: Optional[th.Tensor] = None,
                train_step: bool = True
            )-> Tuple[FeatureMap, LstmState]:
        if self.memory_type in ['lstm', 'gru', 'rnn']:
            return self.forward_rnn(x, h_and_c_previous, token_mask)
        
        elif self.memory_type in ['s4', 's5', 's6']:
            return self.forward_ssm(x, h_and_c_previous, token_mask, train_step)