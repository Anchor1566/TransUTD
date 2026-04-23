"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from torch import Tensor, nn
from typing import List

from .denoising import get_contrastive_denoising_training_group
from .tv_misc import Conv2dNormActivation
from .position_encoding import get_sine_pos_embed
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from ...core import register

from scipy.optimize import linear_sum_assignment

__all__ = ['RTDETRTransformerv2']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GateFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Linear(2, 1)
        self._reset_params()

    def _reset_params(self):
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, y):
        gate = torch.sigmoid(self.gate(torch.stack([x, y], dim=-1))).squeeze(-1)
        return x * gate + y * (1 - gate)


class MSDeformableAttention(nn.Module):
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
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

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

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

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
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
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

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        output = self.output_proj(output)

        return output


def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed

def box_rel_encoding2(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    xy_zero = torch.zeros_like(delta_xy)
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed


class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.0,
        scale=100.0,
        activation_layer=nn.ReLU,
        inplace=True,
        temp = False,
    ):
        super().__init__()
        self.temp = temp
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

    def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        with torch.no_grad():
            if self.temp:
                pos_embed = box_rel_encoding2(src_boxes, tgt_boxes)
                pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
            else:
                pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
                pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        self.num_heads = n_head
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                    target,
                    reference_points,
                    memory,
                    memory_spatial_shapes,
                    ref_target=None,
                    ref_reference_points=None,
                    attn_mask=None,
                    memory_mask=None,
                    query_pos_embed=None, 
                    ref_query_pos_embed=None
                    ):
        # self attention        
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        if ref_target is not None:
            q = k = self.with_pos_embed(ref_target, ref_query_pos_embed)
            ref_target2, _ = self.self_attn(q, k, value=ref_target, attn_mask=attn_mask)
            ref_target = ref_target + self.dropout1(ref_target2)
            ref_target = self.norm1(ref_target)

            # cross attention
            ref_target2 = self.cross_attn(\
                self.with_pos_embed(ref_target, ref_query_pos_embed), 
                ref_reference_points, 
                memory, 
                memory_spatial_shapes, 
                memory_mask)
            ref_target = ref_target + self.dropout2(ref_target2)
            ref_target = self.norm2(ref_target)

            # ffn
            ref_target2 = self.forward_ffn(ref_target)
            ref_target = ref_target + self.dropout4(ref_target2)
            ref_target = self.norm3(ref_target)

            return target, ref_target

        return target, None


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

        self.num_heads = decoder_layer.num_heads
        # relation embedding
        self.position_relation_embedding = PositionRelationEmbedding(32, self.num_heads)
        self.temp_position_relation_embedding = PositionRelationEmbedding(32, self.num_heads)

        self.gatefusion = GateFusion()

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                ref_targets=None,
                ref_ref_points_unact=None,
                attn_mask=None,
                memory_mask=None, 
                skip_relation=False,
                ):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_ref_points_detach = F.sigmoid(ref_ref_points_unact) if ref_ref_points_unact is not None else None

        output = target
        ref_output = ref_targets
        pos_relation = attn_mask
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            if ref_ref_points_detach is not None:
                ref_ref_points_input = ref_ref_points_detach.unsqueeze(2)
                ref_query_pos_embed = query_pos_head(ref_ref_points_detach)
            else:
                ref_ref_points_input = None
                ref_query_pos_embed = None

            output, ref_output = layer(output, ref_points_input, memory, memory_spatial_shapes, ref_output, ref_ref_points_input,
                            attn_mask, memory_mask, query_pos_embed, ref_query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            if ref_output is not None:
                ref_inter_ref_bbox = F.sigmoid(bbox_head[i](ref_output) + inverse_sigmoid(ref_ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            
            if i == self.num_layers - 1:
                break

            if not skip_relation:
                src_boxes = tgt_boxes if i >= 1 else ref_points_detach
                tgt_boxes = inter_ref_bbox
                ref_tgt_boxes = ref_inter_ref_bbox
                pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
                
                ref_pos_relation = self.temp_position_relation_embedding(ref_tgt_boxes, tgt_boxes).flatten(0, 1)
                pos_relation = self.gatefusion(ref_pos_relation, pos_relation)
                if attn_mask is not None:
                    pos_relation.masked_fill_(attn_mask, float("-inf"))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
            if ref_output is not None:
                ref_ref_points_detach = ref_inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register()
class TemporalTransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 batch_size=4,
                 num_refs=3,
                 aux_loss=True, 
                 cross_attn_method='default', 
                 query_select_method='default'):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # TODO
        self.bs = batch_size
        self.num_refs = num_refs
        self.num_groups = num_refs + 1
        
        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
            # if self.num_groups is not None:
            #     self.denoising_class_embed = nn.ModuleList([
            #         nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes) for _ in range(self.num_groups)
            #     ])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries*self.num_groups, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('proj', nn.Linear(hidden_dim, hidden_dim)),
                ('norm', nn.LayerNorm(hidden_dim,)),
            ])) for _ in range(1)
        ])

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(1)
            ])
        else:
            self.enc_score_head = nn.ModuleList([
                nn.Linear(hidden_dim, num_classes) for _ in range(1)
            ])
            
        self.enc_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(1)
        ])

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()
        
    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        for enc_score_head in self.enc_score_head:
            init.constant_(enc_score_head.bias, bias)
        for enc_bbox_head in self.enc_bbox_head:
            init.constant_(enc_bbox_head.layers[-1].weight, 0)
            init.constant_(enc_bbox_head.layers[-1].bias, 0)

        for enc_output in self.enc_output:
            init.xavier_uniform_(enc_output[0].weight)
            init.constant_(enc_output[0].bias, 0)

        # for denoising_class_embed in self.denoising_class_embed:
        #     init.normal_(denoising_class_embed.weight[:-1])

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
        
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # if len(proj_feats) < self.num_levels(predefined levels), add more input_proj
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        """
            Generates anchors at the center of each pixel for multiple feature map levels.
            valid_mask is used to mask out the anchors that are valid in the feature map.
        """
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)  # scale to [0, 1]
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        # spatial_shapes, value: [[80, 80], [40, 40], [20, 20]]
        # anchors, [1, 8400, 4]
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           num_groups=1,
                           ref_num_queries=300,
                           denoising_logits=None,
                           denoising_bbox_unact=None,
                           ):

        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        memory = valid_mask.to(memory.dtype) * memory  
        output_memory :torch.Tensor = self.enc_output[0](memory)  # [bs*T, len, hidden_dim], such as len = 80*80+40*40+20*20, not a fixed number
        enc_outputs_logits :torch.Tensor = self.enc_score_head[0](output_memory)
        # generate offsets for anchors + anchors
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head[0](output_memory) + anchors

        ref_output_memory = output_memory.reshape(self.bs, -1, memory.shape[-1])[:, memory.shape[1]:, :].reshape(self.bs*self.num_refs, -1, memory.shape[-1])
        ref_enc_outputs_logits = enc_outputs_logits.reshape(self.bs, -1, enc_outputs_logits.shape[-1])[:, memory.shape[1]:, :]
        ref_enc_outputs_logits = ref_enc_outputs_logits.reshape(self.bs*self.num_refs, -1, enc_outputs_logits.shape[-1])
        ref_enc_outputs_coord_unact = enc_outputs_coord_unact.reshape(self.bs, -1, enc_outputs_coord_unact.shape[-1])[:, memory.shape[1]:, :]
        ref_enc_outputs_coord_unact = ref_enc_outputs_coord_unact.reshape(self.bs*self.num_refs, -1, enc_outputs_coord_unact.shape[-1])

        cur_output_memory = output_memory[::self.num_groups]
        cur_enc_outputs_logits = enc_outputs_logits[::self.num_groups]
        cur_enc_outputs_coord_unact = enc_outputs_coord_unact[::self.num_groups]

        # enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
        #         self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
        ref_topk_memory, ref_topk_logits, ref_topk_bbox_unact = \
                self._select_topk(ref_output_memory, ref_enc_outputs_logits, ref_enc_outputs_coord_unact, ref_num_queries)
        cur_topk_memory, cur_topk_logits, cur_topk_bbox_unact = \
                self._select_topk(cur_output_memory, cur_enc_outputs_logits, cur_enc_outputs_coord_unact, self.num_queries)

        ref_topk_memory = ref_topk_memory.reshape(self.bs, -1, memory.shape[-1])
        ref_topk_logits = ref_topk_logits.reshape(self.bs, -1, enc_outputs_logits.shape[-1])
        ref_topk_bbox_unact = ref_topk_bbox_unact.reshape(self.bs, -1, enc_outputs_coord_unact.shape[-1])

        ref_topk_rela_memory, ref_topk_rela_bbox_unact = self._select_feats(cur_topk_memory, ref_topk_memory, ref_topk_bbox_unact, ref_topk_logits, 1.00, 0.00)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []

        # o2o for enc_head training
        if self.training:
            cur_enc_topk_bboxes = F.sigmoid(cur_topk_bbox_unact)
            enc_topk_bboxes_list.append(cur_enc_topk_bboxes)
            enc_topk_logits_list.append(cur_topk_logits)

        # FIXME
        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = cur_topk_memory.detach()
            ref_content = ref_topk_memory.detach()
        cur_topk_bbox_unact = cur_topk_bbox_unact.detach()
        ref_topk_bbox_unact = ref_topk_bbox_unact.detach()

        ref_rela_content = ref_topk_rela_memory.detach()
        ref_rela_bbox_unact = ref_topk_rela_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            cur_topk_bbox_unact = torch.concat([denoising_bbox_unact, cur_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
            ref_rela_bbox_unact = torch.concat([denoising_bbox_unact, ref_rela_bbox_unact], dim=1)
            ref_rela_content = torch.concat([denoising_logits, ref_rela_content], dim=1)
        
        return content, cur_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, \
            ref_content, ref_topk_bbox_unact, ref_rela_content, ref_rela_bbox_unact


    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        
        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords
    
    def _cosine_score(self, 
                      cur_feats: torch.Tensor, 
                      ref_feats: torch.Tensor, 
                      ref_logits: torch.Tensor,
                      alpha=0.25,
                      beta=0.75):

        cur_feat_norm = F.normalize(cur_feats, dim=-1)
        ref_feat_norm = F.normalize(ref_feats, dim=-1)

        # caculate the cosine similarity
        cosine_similarity = torch.matmul(cur_feat_norm, ref_feat_norm.transpose(1, 2))

        ref_logits_score = F.softmax(ref_logits, dim=-1)
        ref_logits_score = ref_logits_score.max(-1).values.unsqueeze(-1)
        cosine_score = cosine_similarity.max(1).values.unsqueeze(-1)

        return  cosine_score ** alpha *  ref_logits_score ** beta
    
    
    def _select_feats(self, 
                      cur_feats: torch.Tensor, 
                      ref_feats: torch.Tensor, 
                      ref_coords: torch.Tensor,
                      ref_logits: torch.Tensor,
                      alpha=1,  # logits weight
                      beta=0  # query similarity
                      ):
        bs, _, _ = cur_feats.shape

        matched_feats = []
        matched_coords = []
        for b in range(bs):
            cur_feat = cur_feats[b]
            ref_feat = ref_feats[b]
            ref_coord = ref_coords[b]
            ref_logit = ref_logits[b]

            cur_feat_norm = F.normalize(cur_feat, dim=-1)
            ref_feat_norm = F.normalize(ref_feat, dim=-1)

            # caculate the cosine similarity
            cosine_similarity = torch.matmul(cur_feat_norm, ref_feat_norm.t()).detach()
            cosine_similarity = torch.sigmoid(cosine_similarity)
            cosine_similarity = cosine_similarity / (cosine_similarity.max() + 1e-8)
            # cost_matrix = 1 - cosine_similarity.cpu().numpy()
            ref_logit = torch.sigmoid(ref_logit)
            ref_logit = ref_logit.max(-1).values.unsqueeze(-1)
            ref_logit = ref_logit / (ref_logit.max() + 1e-8)
            cost = -cosine_similarity**alpha * ref_logit.t()**beta
            cost = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_feat = ref_feat[col_ind]
            matched_coord = ref_coord[col_ind]

            matched_feats.append(matched_feat)
            matched_coords.append(matched_coord)

        return torch.stack(matched_feats, dim=0), torch.stack(matched_coords, dim=0)


    def forward(self, feats, targets=None):
        # TODO 
        ref_num_queries = 300
        # input projection and embedding
        # memory: [bs, len, hidden_dim]
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, 
                    num_groups=self.num_groups)

        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        current_memory = memory[::self.num_groups]
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, ref_ref_contents, ref_ref_topk_bbox_unact, \
            ref_rela_contents, ref_rela_points_unact = self._get_decoder_input(memory, spatial_shapes, self.num_groups, ref_num_queries, denoising_logits, denoising_bbox_unact)

        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            current_memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            ref_rela_contents,
            ref_rela_points_unact,
            attn_mask=attn_mask, 
            skip_relation=False,  
            )

        if self.training:
            o2m_out_bboxes, o2m_out_logits = self.decoder(
                ref_ref_contents,
                ref_ref_topk_bbox_unact,
                current_memory,
                spatial_shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head, 
                # attn_mask=attn_mask,
                skip_relation=True,
                )

        if self.training and dn_meta is not None: 
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
            
            if self.num_groups > 1:
                out['pred_logits_one2many'] = o2m_out_logits[-1]
                out['pred_boxes_one2many'] = o2m_out_bboxes[-1]
                out['aux_outputs_one2many'] = self._set_aux_loss(o2m_out_logits[:-1], o2m_out_bboxes[:-1])

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
