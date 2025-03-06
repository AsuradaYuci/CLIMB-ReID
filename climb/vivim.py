from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
import torch
from functools import partial
from mamba.mamba_ssm.modules.mamba_simple import Mamba
# from mamba.mamba_ssm import Mamba
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):  # x : torch.Size([8, 1024, 768])
        x = self.fc1(x)
        # x = self.dwconv(x, nf, H, W)  # torch.Size([8, 3072, 8, 16, 8])
        x = self.act(x)  # torch.Size([8, 1024, 3072])
        x = self.drop(x)  # torch.Size([8, 1024, 3072])
        x = self.fc2(x)  # torch.Size([8, 1024, 768])
        x = self.drop(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # bimamba_type="v3",
            bimamba_type="none",
            nframes=8
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ln_2 = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B, C, nf, H, W = x.shape  # torch.Size([8, 768, 8, 16, 8])
        #
        # assert C == self.dim
        # n_tokens = x.shape[2:].numel()  # 8 * 16*8 = 1024
        # img_dims = x.shape[2:]  # torch.Size([8, 16, 8])
        # x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)   # b, num_token, D = b, thw, 768  torch.Size([8, 1024, 768])

        x_mamba = x + self.drop_path(self.mamba(self.norm1(x)))  # torch.Size([8, 1024, 768])
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba)))  # torch.Size([8, 1024, 768])
        # out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)  #
        x_mamba = self.ln_2(x_mamba)  # torch.Size([30, 1024, 768])
        return x_mamba
