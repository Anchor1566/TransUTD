import torch
import torch.nn as nn
import torch.nn.functional as F


# Hack implementation 
def set_value(cfg, args, keys=None):
    # for hack implementation, they must be equal
    # reset and unify the value
    default_keys = {
        'num_refs': [
            'TemporalFusionEncoder.num_refs',
            'TemporalTransformer.num_refs', 
            'val_multi_dataloader.dataset.num_refs',
            'train_multi_dataloader.dataset.num_refs',
            'TemporalCriterion.num_refs'
        ],
        'batch_size': [
            'TemporalFusionEncoder.batch_size',
            'TemporalTransformer.batch_size', 
            'val_multi_dataloader.total_batch_size',
            'train_multi_dataloader.total_batch_size'
        ]
    }
    
    if keys is None:
        keys = default_keys
    else:
        keys = {k: keys.get(k, default_keys.get(k, []) ) for k in set(list(keys.keys()) + list(default_keys.keys()))}
    
    if hasattr(args, 'num_refs'):
        for key_path in keys.get('num_refs', []):
            _set_nested_dict_value(cfg.yaml_cfg, key_path, args.num_refs)
     
    if hasattr(args, 'batch_size'):
        for key_path in keys.get('batch_size', []):
            _set_nested_dict_value(cfg.yaml_cfg, key_path, args.batch_size)
    
    return cfg

def _set_nested_dict_value(d, key_path, value):
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


class Gate(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.droupout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.linear2.weight, 0.)
        nn.init.constant_(self.linear2.bias, 0.)

    def ffn(self, x):
        return self.droupout(self.activation(self.linear1(x)))
    
    def forward(self, q, v):
        qv = torch.cat((q, v), dim=-1)
        gate = self.sigmoid(self.ffn(qv))
        return (1 - gate) * q + gate * v


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''
    def __init__(self, 
                 c_in: int, 
                 c_out: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int, 
                 dilation: int,
                 use_relu: bool=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, 
                c_out, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, 
                 c: int, 
                 c_mid: int=64, 
                 scale: int=2, 
                 k_up: int=5, 
                 k_enc: int=3):
        """ 
        The details are in "https://arxiv.org/abs/1905.02188".

        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1, 
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale*k_up)**2, kernel_size=k_enc, 
                              stride=1, padding=k_enc//2, dilation=1, 
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)

    def forward(self, x) -> torch.Tensor:
        b, c, h, w = x.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(x)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)                         # b * 25 * h_ * w_

        x = self.upsmp(x)                               # b * c * h_ * w_
        x = self.unfold(x)                              # b * 25c * h_ * w_
        x = x.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_

        x = torch.einsum('bkhw,bckhw->bchw', [W, x])    # b * c * h_ * w_
        return x
    
# Dysample
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

