from .backbone import *


class corr3D(nn.Module):

    def __init__(self, dim, R, window_size):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        self.window_size = window_size

        #self.qk = nn.Linear(dim, dim * 2, bias=False)
        self.projq = nn.Linear(dim, dim)
        self.projk = nn.Linear(dim, dim)
        self.projv = nn.Linear(dim, dim)
        self.reduceR = nn.Linear(R * dim, dim)
        self.drop = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim=-1)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), R))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.zeros(self.window_size[0]).long()
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, x_ref, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        Nw, R, N, C = x_ref.shape
        x = self.projv(x)
        x_ref = (self.projq(x_ref[:,0:1,:,:]) * self.scale) @ (self.projk(x_ref).permute(0,1,3,2).contiguous()) # B*nW, R, Wd*Wh*Ww, Wd*Wh*Ww
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        x_ref = x_ref + relative_position_bias.unsqueeze(0)
        x_ref = self.softmax(x_ref) # B*nW, R, Wd*Wh*Ww, Wd*Wh*Ww
        x = x_ref @ x   # B*nW, R, Wd*Wh*Ww, Wd*Wh*Ww, C
        x = x.permute(0, 2, 1, 3).contiguous().view(Nw, N, R * C)
        x = self.drop(self.reduceR(x))
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, R, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, R, D, H, W, C = x.shape
    x = x.view(B, R, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7, 8).contiguous().view(B * (D // window_size[0]) * (H // window_size[1]) * (W // window_size[1]), R, window_size[0] * window_size[1] * window_size[2], C)
    return windows

def window_reverse(windows, window_size, B, R, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], R, window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7, 8).contiguous().view(B, R, D, H, W, -1)
    return x

class corrBlock3D(nn.Module):

    def __init__(self, dim, R, window_size=(2,7,7), shift_size=(0,0,0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.attn = corr3D(dim, R, window_size=self.window_size)

    def forward(self, x, x_ref, mask_matrix):
        B, R, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # cyclic shift
    
        # partition windows
        x_windows = window_partition(x, window_size)  # B*nW, R, Wd*Wh*Ww, C
        x_ref_windows = window_partition(x_ref, window_size)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, x_ref_windows, mask=None)  # B*nW, R, Wd*Wh*Ww, C
        # merge windows
        x = window_reverse(attn_windows, window_size, B, 1, D, H, W)
        # reverse cyclic shift

        return x

class corr_BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 in_dim,
                 R,
                 window_size=(1,7,7)):
        super().__init__()
        self.window_size = window_size

        # build blocks
        self.blk = corrBlock3D(
                dim=in_dim,
                R = R,
                window_size=window_size,
                shift_size=(0,0,0)
            )
        

    def forward(self, x, x_ref):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, R, D, C, H, W).
        """
        # calculate attention mask for SW-MSA
        x = x.permute(0, 1, 2, 4, 5, 3).contiguous()
        x_ref = x_ref.permute(0, 1, 2, 4, 5, 3).contiguous()
        x = self.blk(x, x_ref, None)
        return x