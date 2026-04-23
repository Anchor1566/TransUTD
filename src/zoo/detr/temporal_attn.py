import torch 
import functools
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import math

from typing import List
from .utils import get_activation


class TemporalAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        self.num_points = num_points

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.temp_attn = SampleAttention(self.head_dim)
        self.gate = Gate(self.head_dim)
        self.temporal_attention_core = functools.partial(temporal_attention_core, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = []
            for shape, num_points in zip(value_spatial_shapes, self.num_points_list):
                repeated_shapes = [shape] * num_points
                offset_normalizer.extend(repeated_shapes)
            offset_normalizer = torch.tensor(offset_normalizer).flip(1).to(reference_points.device) # sum(self.num_points_list), 2
            sampling_offsets = sampling_offsets / offset_normalizer
            sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, self.num_levels, self.num_points,2)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets
            sampling_locations = sampling_locations.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.temporal_attention_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list, self.temp_attn, self.gate)

        output = self.output_proj(output)

        return output


def temporal_attention_core(\
    value: torch.Tensor, 
    value_spatial_shapes,
    sampling_locations: torch.Tensor, 
    attention_weights: torch.Tensor, 
    num_points_list: List[int], 
    temp_attn: nn.Module,
    gate: nn.Module,
    method='default'):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape
        
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    # sampling_offsets [8, 480, 8, 12, 2]
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1

    elif method == 'discrete':
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == 'default':
            sampling_value_l = F.grid_sample(
                value_l, 
                sampling_grid_l, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False)
        
        elif method == 'discrete':
            # n * m, seq, n, 2
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5).to(torch.int64)

            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1) 
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2) 

            s_idx = torch.arange(sampling_coord.shape[0], device=value.device).unsqueeze(-1).repeat(1, sampling_coord.shape[1])
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]] # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])
        
        sampling_value_list.append(sampling_value_l)
    
    sampling_value = torch.cat(sampling_value_list, dim=-1)  # [bs*8, 32, Len_q, sum_points]
    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))  
    temp_weights = temp_attn(sampling_value * attn_weights)
    # weighted_sample_locs = sampling_value * attn_weights
    weighted_sample_locs = sampling_value *  F.softmax(attn_weights + temp_weights, dim=-1)
    
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


class SampleAttention(nn.Module):
    def __init__(self, channels=32, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels//reduction),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, kernel_size=1, stride=1),
            nn.Identity(),
        )

        self.temp_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x[bs*num_heads, C//num_heads, len, sum_points]
        # x[32, 32, 400*4, 32]
        # ECA-Net like
        ch = self.sigmoid(self.channel_attn(x))
        x = ch * x
        # sp = self.spatial_attn(x)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([x_avg, x_max], dim=1)
        temp = self.sigmoid(self.temp_attn(x))
        return ch * temp 
    

class Gate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(2 * hidden_dim, hidden_dim, 1)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def ffn(self, x):
        return self.activation(self.norm(self.conv1(x)))
    
    def forward(self, q, v):
        qv = torch.cat((q, v), dim=1)
        gate = self.sigmoid(self.ffn(qv))
        return (1 - gate) * q + gate * v