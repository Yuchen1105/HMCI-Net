                      
                       
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import os
import sys
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
from typing import Tuple
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from mamba_ssm import Mamba
from einops import rearrange
import math
from typing import Optional

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

DEFAULT_LOGFILE_LEVEL = 'debug'
DEFAULT_STDOUT_LEVEL = 'info'
DEFAULT_LOG_FILE = './default.log'
DEFAULT_LOG_FORMAT = '%(asctime)s %(levelname)-7s %(message)s'

LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


                                                                                                                                         

class LayerNorm(nn.Module):
    ''

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
                                         
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class OScanner:
    def __init__(self, slice):

        self.onion_scan_indices = self._generate_onion_scan_indices(slice, slice, slice)

    def _snake_scan_2d(self, height, width, reverse=False):
        positions = []
        for i in range(height):
            if (i % 2 == 0 and not reverse) or (i % 2 != 0 and reverse):
                for j in range(width):
                    positions.append((i, j))
            else:
                for j in range(width - 1, -1, -1):
                    positions.append((i, j))
        return positions

    def _perimeter_scan_2d(self, height, width, start_position=None, counter_clockwise=False):
        positions = []
        if not counter_clockwise:
            for j in range(width):
                positions.append((0, j))
            for i in range(1, height):
                positions.append((i, width - 1))
            for j in range(width - 2, -1, -1):
                positions.append((height - 1, j))
            for i in range(height - 2, 0, -1):
                positions.append((i, 0))
        else:
            for j in range(width - 1, -1, -1):
                positions.append((0, j))
            for i in range(1, height):
                positions.append((i, 0))
            for j in range(1, width):
                positions.append((height - 1, j))
            for i in range(height - 2, 0, -1):
                positions.append((i, width - 1))

        if start_position:
            start_i, start_j = start_position
            start_idx = None
            for idx, (i, j) in enumerate(positions):
                if i == start_i and j == start_j:
                    start_idx = idx
                    break
            if start_idx is not None:
                positions = positions[start_idx:] + positions[:start_idx]

        return positions

    def _ring_scan_2d(self, height, width):
        positions = []
        top, bottom = 0, height - 1
        left, right = 0, width - 1

        while top <= bottom and left <= right:
            if top == bottom:
                for j in range(left, right + 1):
                    positions.append((top, j))
                break
            if left == right:
                for i in range(top, bottom + 1):
                    positions.append((i, left))
                break

            for j in range(left, right + 1):
                positions.append((top, j))
            for i in range(top + 1, bottom + 1):
                positions.append((i, right))
            for j in range(right - 1, left - 1, -1):
                positions.append((bottom, j))
            for i in range(bottom - 1, top, -1):
                positions.append((i, left))

            top += 1
            bottom -= 1
            left += 1
            right -= 1

        return positions

    def _generate_onion_scan_indices(self, D, H, W):
        all_indices = []
        layer_positions = self._ring_scan_2d(H, W)                          

        for z in range(D):
            pos = layer_positions if (z % 2 == 0) else list(reversed(layer_positions))
            for i, j in pos:
                all_indices.append([z, i, j])

        return torch.tensor(all_indices, dtype=torch.long)

    def scan_tensor(self, tensor, indices=None):
        if indices is None:
            indices = self.onion_scan_indices.to(tensor.device)
        d_idx, h_idx, w_idx = indices[:, 0], indices[:, 1], indices[:, 2]
        out = tensor[:, :, d_idx, h_idx, w_idx]                                      
        return out

    def restore_tensor(self, seq_tensor, D, H, W, indices=None):
        B, C, _ = seq_tensor.shape
        device = seq_tensor.device

        if indices is None:
            indices = self.onion_scan_indices.to(device)
        else:
            indices = indices.to(device)

        flat_idx = indices[:, 0] * (H * W) + indices[:, 1] * W + indices[:, 2]            

        out = torch.zeros((B, C, D * H * W), dtype=seq_tensor.dtype, device=device)
        out.scatter_(2, flat_idx.unsqueeze(0).unsqueeze(0).expand(B, C, -1), seq_tensor)
        out = out.view(B, C, D, H, W)
        return out


class ux_block(nn.Module):


    def __init__(self, dim, slice, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

                                                                                                
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv3d(dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mamba1 = Mamba(
            d_model=dim // 3,                           
            d_state=16,                              
            d_conv=4,                           
            expand=2,                          
        )
        self.mamba2 = Mamba(
            d_model=dim // 3,                           
            d_state=16,                              
            d_conv=4,                           
            expand=2,                          
        )
        self.mamba3 = Mamba(
            d_model=dim // 3,                           
            d_state=16,                              
            d_conv=4,                           
            expand=2,                          
        )
        self.Scan = OScanner(slice)
        self.indices = self.Scan.onion_scan_indices
        self.AMFWFusion = TripleBranchFusion_KAN(in_channels=dim, out_channels=dim)


    def forward(self, x):
                                                           
        input = x
        Res = x
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm1(x)
        x = x.permute(0, 4, 1, 2, 3)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
                                                                
                                                                  
        seq_x1 = self.Scan.scan_tensor(rearrange(x1, "b c w h d -> b c w h d"))
        seq_x1_reverse = torch.flip(seq_x1, dims=[2])
                                                                          
        seq_x1 = self.mamba1(seq_x1.permute(0, 2, 1))
        seq_x1_reverse = self.mamba1(seq_x1_reverse.permute(0, 2, 1))
        seq_x1_branch = (seq_x1 + torch.flip(seq_x1_reverse, dims=[1])) / 2
                                                                                      
        x1 = self.Scan.restore_tensor(seq_x1_branch.permute(0, 2, 1), D, H, W)
                                                                           
        x1 = (rearrange(x1, "b c w h d -> b c w h d"))
                                                                                 

                                                                
                                                                  
        seq_x2 = self.Scan.scan_tensor(rearrange(x2, "b c w h d -> b c d w h"))
        seq_x2_reverse = torch.flip(seq_x2, dims=[2])
                                                                          
        seq_x2 = self.mamba2(seq_x2.permute(0, 2, 1))
        seq_x2_reverse = self.mamba2(seq_x2_reverse.permute(0, 2, 1))
        seq_x2_branch = (seq_x2 + torch.flip(seq_x2_reverse, dims=[1])) / 2
                                                                                      
        x2 = self.Scan.restore_tensor(seq_x2_branch.permute(0, 2, 1), D, H, W)
                                                                           
        x2 = (rearrange(x2, "b c d w h -> b c w h d"))
                                                                                 

                                                                
                                                                  
        seq_x3 = self.Scan.scan_tensor(rearrange(x3, "b c w h d -> b c d w h"))
        seq_x3_reverse = torch.flip(seq_x3, dims=[2])
                                                                          
        seq_x3 = self.mamba3(seq_x3.permute(0, 2, 1))
        seq_x3_reverse = self.mamba3(seq_x3_reverse.permute(0, 2, 1))
        seq_x3_branch = (seq_x3 + torch.flip(seq_x3_reverse, dims=[1])) / 2
                                                                                      
        x3 = self.Scan.restore_tensor(seq_x3_branch.permute(0, 2, 1), D, H, W)
                                                                           
        x3 = (rearrange(x3, "b c d w h -> b c w h d"))
                                                                                 

        x = torch.cat([x1, x2, x3], dim=1)
        x = x + Res

        x = self.AMFWFusion(x)
                                                            
        x = x.permute(0, 2, 3, 4, 1)                                      
                                                              
        x = self.norm2(x)
                                                               
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
                                                               
        x = input + self.drop_path(x)
                                                            
        return x


class TripleBranchFusion_KAN(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(TripleBranchFusion_KAN, self).__init__()

                                                                  
        self.volume_extractor = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=in_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=1)
        )
        self.feature_extractor = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)

                                
        self.gap1 = nn.AdaptiveAvgPool3d(1)
        self.gap2 = nn.AdaptiveMaxPool3d(1)

                                                              
        self.kan_alpha_1 = KANLinear(out_channels, out_channels // reduction)
        self.kan_alpha_2 = KANLinear(out_channels // reduction, out_channels)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, F_i):
                                   
        F_res = F_i
        F_v = self.volume_extractor(F_i)
                                          

                                  
        F_g = (self.gap1(F_v) + self.gap2(F_v)) / 2                   
        F_g = F_g.view(F_g.size(0), F_g.size(1))          
                                                                   

                                           
        alpha = self.kan_alpha_2(self.kan_alpha_1(F_g))
                                                        

                                                
        alpha = self.softmax(alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                   
                                                            

                                            
        F_o = F_v * alpha
                                                                   

                                         
        F_i = self.feature_extractor(F_i)
                                                                   

                                                                                     
        return F_o * F_i + F_res

class TripleBranchFusion_KAN_V2(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, groups_gn=8):
        super().__init__()
        mid_channels = out_channels

                  
        self.extractor1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=5, padding=2, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=min(groups_gn, mid_channels), num_channels=mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(groups_gn, mid_channels), num_channels=mid_channels),
            nn.GELU(),
        )

                                    
        self.extractor2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=min(groups_gn, in_channels), num_channels=in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(groups_gn, mid_channels), num_channels=mid_channels),
            nn.GELU(),
        )

                         
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)

                   
        red = max(1, mid_channels // reduction)
        self.kan_alpha_1 = KANLinear(mid_channels, red)
        self.kan_alpha_2 = KANLinear(red, mid_channels)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

              
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, F_i):
        F_res = F_i

        F_v = self.extractor1(F_i)

                      
        avg = self.gap(F_v).flatten(1)
        mx  = self.gmp(F_v).flatten(1)
        F_g = 0.5 * (avg + mx)

                           
        alpha = self.sigmoid(self.kan_alpha_2(self.act(self.kan_alpha_1(F_g))))
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        F_o = F_v * alpha
        F_i_feat = self.extractor2(F_i)

        out = F_res + self.gamma * (F_o * F_i_feat)
        return out

class uxnet_conv(nn.Module):       
    """"""

    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()                                                    
                               
                                                                               
                                                                        
           
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        slice = [64, 32, 16, 8]
        for i in range(4):
            stage = nn.Sequential(
                *[ux_block(dim=dims[i], slice=slice[i], drop_path=dp_rates[cur + j],
                           layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

                                        

    def forward_features(self, x):
        outs = []
        for i in range(4):
                        
                               
            x = self.downsample_layers[i](x)
                               
            x = self.stages[i](x)
                               
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Logger.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


                           
                          
                           
class LayerNorm3d(nn.Module):
    """"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
                                            
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


                           
                    
                           
class HLKConv3D(nn.Module):
    """"""
    def __init__(self, dim, k_size=23):
        super().__init__()
        self.k_size = k_size

        if k_size == 7:
            k0, ks, d = 3, 3, 2
        elif k_size == 11:
            k0, ks, d = 3, 5, 2
        elif k_size == 23:
            k0, ks, d = 5, 7, 3
        elif k_size == 35:
            k0, ks, d = 5, 11, 3
        elif k_size == 41:
            k0, ks, d = 5, 13, 3
        elif k_size == 53:
            k0, ks, d = 5, 17, 3
        else:
            raise ValueError(f"Unsupported k_size: {k_size}. Choose from [7, 11, 23, 35, 41, 53].")

        p0 = (k0 - 1) // 2
        ps = d * (ks - 1) // 2

        self.conv0 = nn.Conv3d(
            dim, dim, kernel_size=k0, stride=1, padding=p0,
            groups=dim, bias=False
        )
        self.conv_spatial = nn.Conv3d(
            dim, dim, kernel_size=ks, stride=1, padding=ps,
            dilation=d, groups=dim, bias=False
        )
        self.conv1 = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.conv0(x)
        xs = self.conv_spatial(x0)
        out = self.conv1(torch.cat([x0, xs], dim=1))
        return out


                           
           
                           
class KANLinear(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2
                )
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output


                           
                             
                           
class KANChannelAttention3D(nn.Module):
    """"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_dim = max(1, channels // reduction)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)

        self.norm = nn.LayerNorm(channels)
        self.kan1 = KANLinear(channels, hidden_dim)
        self.kan2 = KANLinear(hidden_dim, channels)

        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.gap(x).flatten(1)           
        mx  = self.gmp(x).flatten(1)           

        g = 0.5 * (avg + mx)
        g = self.norm(g)

        attn = self.kan2(self.act(self.kan1(g)))
        attn = self.sigmoid(attn).view(x.size(0), x.size(1), 1, 1, 1)
        return attn


                           
                                  
                           
class SpatialAttentionHLK3D(nn.Module):
    """"""
    def __init__(self, hlk_size=23):
        super().__init__()
        self.hlk = HLKConv3D(dim=2, k_size=hlk_size)
        self.proj = nn.Conv3d(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg_out, max_out], dim=1)
        s = self.hlk(s)
        s = self.proj(s)
        return self.sigmoid(s)


                           
                                                         
                           
class AGFF3D_KAN_HLK_Bottleneck(nn.Module):
    """"""
    def __init__(self, in_channels, reduction=4, hlk_size=23):
        super().__init__()
        out_channels = in_channels * 2
        self.in_channels = in_channels
        self.out_channels = out_channels

                                      
        self.identity_proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm3d(out_channels)
        )

                                             
        self.context_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False),
            LayerNorm3d(in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm3d(out_channels),
            nn.GELU()
        )

                                       
        self.local_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            LayerNorm3d(in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm3d(out_channels),
            nn.GELU()
        )

                                             
        self.ca_context = KANChannelAttention3D(out_channels, reduction=reduction)
        self.ca_local   = KANChannelAttention3D(out_channels, reduction=reduction)
        self.ca_global  = KANChannelAttention3D(out_channels, reduction=reduction)

                           
        self.sa = SpatialAttentionHLK3D(hlk_size=hlk_size)

                                      
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm3d(out_channels),
            nn.GELU()
        )

                          
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = self.identity_proj(x)                     

        x_context = self.context_branch(x)                   
        x_local   = self.local_branch(x)                     

                                    
        w_context = self.ca_context(x_context)
        w_local   = self.ca_local(x_local)

        denom = w_context + w_local + 1e-6
        alpha_context = w_context / denom
        alpha_local   = w_local / denom

                             
        w_global = self.ca_global(x_context + x_local)
        alpha_context = alpha_context * w_global
        alpha_local   = alpha_local * w_global

                       
        x_fused = alpha_context * x_context + alpha_local * x_local

                             
        s_attn = self.sa(x_fused)                                      
        x_fused = self.refine(x_fused) * s_attn                        

                         
        out = identity + self.gamma * x_fused
        return out

class HPA3D(nn.Module):
    """"""

    def __init__(self, channels, c2=None, factor=32):
        super(HPA3D, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0, "channels must be divisible by groups (factor)"
        self.c = channels
        self.cg = channels // self.groups                      
        self.softmax = nn.Softmax(dim=-1)

                                         
        self.agp3 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.map3 = nn.AdaptiveMaxPool3d((1, 1, 1))

                                                                                              
                                                                                                                 
        self.conv1x1 = nn.Conv2d(self.cg, self.cg, kernel_size=1, stride=1, padding=0, bias=True)
                                       
        self.conv3x3x3 = nn.Conv3d(self.cg, self.cg, kernel_size=3, stride=1, padding=1, bias=True)

                                                                    
                                                                                 
                                                     
        self.pwcconv = nn.Conv3d(channels, channels * 2, kernel_size=1, groups=channels)
        self.gn = nn.GroupNorm(1, self.cg)

    def _directional_pools(self, g):
        """"""
                                 
        x_d = g.mean(dim=(3, 4), keepdim=True).squeeze(-1).squeeze(-1)                
        x_d = x_d.unsqueeze(-1)                   

                                 
        x_h = g.mean(dim=(2, 4), keepdim=True).squeeze(-1).squeeze(2)                
        x_h = x_h.unsqueeze(-1)                   

                                 
        x_w = g.mean(dim=(2, 3), keepdim=True).squeeze(2).squeeze(2)                
        x_w = x_w.unsqueeze(-1)                   

        return x_d, x_h, x_w

    def _directional_maxpools(self, g):
        """"""
        y_d = torch.amax(g, dim=(3, 4), keepdim=True).squeeze(-1).squeeze(-1).unsqueeze(-1)
        y_h = torch.amax(g, dim=(2, 4), keepdim=True).squeeze(-1).squeeze(2).unsqueeze(-1)
        y_w = torch.amax(g, dim=(2, 3), keepdim=True).squeeze(2).squeeze(2).unsqueeze(-1)
        return y_d, y_h, y_w

    def forward(self, x):
        """"""
        b, c, d, h, w = x.shape
        assert c == self.c

                                                      
        group_x = x.reshape(b * self.groups, self.cg, d, h, w)

                             
                                  
                             
        xd, xh, xw = self._directional_pools(group_x)                              
                                                                  
        cat = torch.cat([xd, xh, xw], dim=2)                                   
                                                                                    
        hw = self.conv1x1(cat)                   
                                                    
        ld, lh, lw = d, h, w
        x_d2, x_h2, x_w2 = torch.split(hw, [ld, lh, lw], dim=2)
                                                               
        x_d2 = x_d2.view(b * self.groups, self.cg, d, 1, 1)                    
        x_h2 = x_h2.view(b * self.groups, self.cg, 1, h, 1)                    
        x_w2 = x_w2.view(b * self.groups, self.cg, 1, 1, w)                    

        x1 = self.gn(group_x * torch.sigmoid(x_d2) * torch.sigmoid(x_h2) * torch.sigmoid(x_w2))


                             
                                  
                             
        yd, yh, yw = self._directional_maxpools(group_x)
        cat_y = torch.cat([yd, yh, yw], dim=2)                   
        yhw = self.conv1x1(cat_y)
        y_d2, y_h2, y_w2 = torch.split(yhw, [ld, lh, lw], dim=2)
        y_d2 = y_d2.view(b * self.groups, self.cg, d, 1, 1)
        y_h2 = y_h2.view(b * self.groups, self.cg, 1, h, 1)
        y_w2 = y_w2.view(b * self.groups, self.cg, 1, 1, w)

        y1 = self.gn(group_x * torch.sigmoid(y_d2) * torch.sigmoid(y_h2) * torch.sigmoid(y_w2))

                             
                                                                                      
                             
        n = d * h * w
        y11 = y1.reshape(b * self.groups, self.cg, -1)                
                                                                                                         
        y12 = self.softmax(self.map3(y1).reshape(b * self.groups, self.cg, 1).permute(0, 2, 1))              

        x11 = x1.reshape(b * self.groups, self.cg, -1)                
        x12 = self.softmax(self.agp3(x1).reshape(b * self.groups, self.cg, 1).permute(0, 2, 1))              


                                                                        
        part1 = torch.matmul(x12, y11)             
        part2 = torch.matmul(y12, x11)             
        weights = (part1 + part2).reshape(b * self.groups, 1, d, h, w)                 

        out = (group_x * torch.sigmoid(weights)).reshape(b, c, d, h, w)
        out = self.pwcconv(out)
        return out


class UXNET(nn.Module):

    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
                                          
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        '''self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )'''
        self.encoder5 = HPA3D(channels=self.feat_size[3])

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
                                                             

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)

        enc1 = self.encoder1(x_in)
                                             

        x2 = outs[0]
                                                   
        enc2 = self.encoder2(x2)
                                             

        x3 = outs[1]
                                                   
        enc3 = self.encoder3(x3)
                                             

        x4 = outs[2]
                                                   
        enc4 = self.encoder4(x4)
                                             

                                                                          
        enc_hidden = self.encoder5(outs[3])
                                                                                

        dec3 = self.decoder5(enc_hidden, enc4)                       
                                             
        dec2 = self.decoder4(dec3, enc3)
                                             
        dec1 = self.decoder3(dec2, enc2)
                                             
        dec0 = self.decoder2(dec1, enc1)
                                             
        out = self.decoder1(dec0)
                                           

                                     

        return self.out(out)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UXNET(
        in_chans=1,
        out_chans=8,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3)
    input_tensor = torch.randn(1, 1, 128, 128, 128).to(device)
    model.to(device=device)
    images = input_tensor
    res1 = model(images)
    '''from thop import profile, clever_format
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")# print(f"Parameters: {params}")# print(f"FLOPs: {flops}")'''


                                                                                                                          

class Logger(object):
    """"""
    logfile_level = None
    log_file = None
    log_format = None
    rewrite = None
    stdout_level = None
    logger = None

    _caches = {}

    @staticmethod
    def init(logfile_level=DEFAULT_LOGFILE_LEVEL,
             log_file=DEFAULT_LOG_FILE,
             log_format=DEFAULT_LOG_FORMAT,
             rewrite=False,
             stdout_level=None):
        Logger.logfile_level = logfile_level
        Logger.log_file = log_file
        Logger.log_format = log_format
        Logger.rewrite = rewrite
        Logger.stdout_level = stdout_level

        Logger.logger = logging.getLogger()
        Logger.logger.handlers = []
        fmt = logging.Formatter(Logger.log_format)

        if Logger.logfile_level is not None:
            filemode = 'w'
            if not Logger.rewrite:
                filemode = 'a'

            dir_name = os.path.dirname(os.path.abspath(Logger.log_file))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            if Logger.logfile_level not in LOG_LEVEL_DICT:
                                                                                 
                Logger.logfile_level = DEFAULT_LOGFILE_LEVEL

            Logger.logger.setLevel(LOG_LEVEL_DICT[Logger.logfile_level])

            fh = logging.FileHandler(Logger.log_file, mode=filemode)
            fh.setFormatter(fmt)
            fh.setLevel(LOG_LEVEL_DICT[Logger.logfile_level])

            Logger.logger.addHandler(fh)

        if stdout_level is not None:
            if Logger.logfile_level is None:
                Logger.logger.setLevel(LOG_LEVEL_DICT[Logger.stdout_level])

            console = logging.StreamHandler()
            if Logger.stdout_level not in LOG_LEVEL_DICT:
                                                                                
                return

            console.setLevel(LOG_LEVEL_DICT[Logger.stdout_level])
            console.setFormatter(fmt)
            Logger.logger.addHandler(console)

    @staticmethod
    def set_log_file(file_path):
        Logger.log_file = file_path
        Logger.init(log_file=file_path)

    @staticmethod
    def set_logfile_level(log_level):
        if log_level not in LOG_LEVEL_DICT:
                                                                  
            return

        Logger.init(logfile_level=log_level)

    @staticmethod
    def clear_log_file():
        Logger.rewrite = True
        Logger.init(rewrite=True)

    @staticmethod
    def check_logger():
        if Logger.logger is None:
            Logger.init(logfile_level=None, stdout_level=DEFAULT_STDOUT_LEVEL)

    @staticmethod
    def set_stdout_level(log_level):
        if log_level not in LOG_LEVEL_DICT:
                                                                  
            return

        Logger.init(stdout_level=log_level)

    @staticmethod
    def debug(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        Logger.logger.debug('{} {}'.format(prefix, message))

    @staticmethod
    def info(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        Logger.logger.info('{} {}'.format(prefix, message))

    @staticmethod
    def info_once(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)

        if Logger._caches.get((prefix, message)) is not None:
            return

        Logger.logger.info('{} {}'.format(prefix, message))
        Logger._caches[(prefix, message)] = True

    @staticmethod
    def warn(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        Logger.logger.warn('{} {}'.format(prefix, message))

    @staticmethod
    def error(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        Logger.logger.error('{} {}'.format(prefix, message))

    @staticmethod
    def critical(message):
        Logger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        Logger.logger.critical('{} {}'.format(prefix, message))


class ModuleHelper(object):

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == 'torchbn':
            return nn.Sequential(
                nn.BatchNorm3d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'torchsyncbn':
            return nn.Sequential(
                nn.SyncBatchNorm(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return nn.Sequential(
                BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return nn.Sequential(
                SwitchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'gn':
            return nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'fn':
            Logger.error('Not support Filter-Response-Normalization: {}.'.format(bn_type))
            exit(1)
        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
                                                                  
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver in ('1.0', '1.1'):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)

        else:
            Logger.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type='torch', ret_cls=False):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d

        elif bn_type == 'torchsyncbn':
            return nn.SyncBatchNorm

        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return BatchNorm2d

        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return SwitchNorm2d

        elif bn_type == 'gn':
            return functools.partial(nn.GroupNorm, num_groups=32)

        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='none')

            elif torch_ver in ('1.0', '1.1'):
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='none')

            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync

                return functools.partial(InPlaceABNSync, activation='identity')

        else:
            Logger.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network='resnet101'):
        if pretrained is None:
            return model

        if all_match:
            Logger.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'resinit.{}'.format(k) in model_dict:
                    load_dict['resinit.{}'.format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)

        else:
            Logger.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)

                                                                     
            if network == "wide_resnet":
                pretrained_dict = pretrained_dict['state_dict']

            model_dict = model.state_dict()

            if network == "hrnet_plus":
                                                                                            
                                                                                            
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pvt':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                pretrained_dict['pos_embed1'] =\
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed2'] =\
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed3'] =\
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                pretrained_dict['pos_embed4'] =\
                    interpolate(pretrained_dict['pos_embed4'].unsqueeze(dim=0), size=[256, 512])[0]
                pretrained_dict['pos_embed7'] =\
                    interpolate(pretrained_dict['pos_embed1'].unsqueeze(dim=0), size=[16384, 64])[0]
                pretrained_dict['pos_embed6'] =\
                    interpolate(pretrained_dict['pos_embed2'].unsqueeze(dim=0), size=[4096, 128])[0]
                pretrained_dict['pos_embed5'] =\
                    interpolate(pretrained_dict['pos_embed3'].unsqueeze(dim=0), size=[1024, 320])[0]
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

            elif network == 'pcpvt' or network == 'svt':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                Logger.info('Missing keys: {}'.format(list(set(model_dict) - set(load_dict))))

            elif network == 'transunet_swin':
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict.keys()}
                for item in list(pretrained_dict.keys()):
                    if item.startswith('layers.0') and not item.startswith('layers.0.downsample'):
                        pretrained_dict['dec_layers.2' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.1') and not item.startswith('layers.1.downsample'):
                        pretrained_dict['dec_layers.1' + item[15:]] = pretrained_dict[item]
                    if item.startswith('layers.2') and not item.startswith('layers.2.downsample'):
                        pretrained_dict['dec_layers.0' + item[15:]] = pretrained_dict[item]

                for item in list(pretrained_dict.keys()):
                    if 'relative_position_index' in item:
                        pretrained_dict[item] =\
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[256, 256])[0][0]
                    if 'relative_position_bias_table' in item:
                        pretrained_dict[item] =\
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[961, pretrained_dict[item].size(1)])[0][0]
                    if 'attn_mask' in item:
                        pretrained_dict[item] =\
                            interpolate(pretrained_dict[item].unsqueeze(dim=0).unsqueeze(dim=0).float(),
                                        size=[pretrained_dict[item].size(0), 256, 256])[0][0]

            elif network == "hrnet" or network == "xception" or network == 'resnest':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                Logger.info('Missing keys: {}'.format(list(set(model_dict) - set(load_dict))))

            elif network == "dcnet" or network == "resnext":
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if 'resinit.{}'.format(k) in model_dict:
                        load_dict['resinit.{}'.format(k)] = v
                    else:
                        if k in model_dict:
                            load_dict[k] = v
                        else:
                            pass

            elif network == "wide_resnet":
                load_dict = {'.'.join(k.split('.')[1:]): v\
                             for k, v in pretrained_dict.items()\
                             if '.'.join(k.split('.')[1:]) in model_dict}
            else:
                load_dict = {'.'.join(k.split('.')[1:]): v\
                             for k, v in pretrained_dict.items()\
                             if '.'.join(k.split('.')[1:]) in model_dict}

                           
            if int(os.environ.get("debug_load_model", 0)):
                Logger.info('Matched Keys List:')
                for key in load_dict.keys():
                    Logger.info('{}'.format(key))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.PyTorchCV', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)

        Logger.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
