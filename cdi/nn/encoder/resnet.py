import warnings

from collections import defaultdict
from itertools import repeat

import torch
from torch import nn

from cdi.bundled.leanbase.torch.select import select_norm_layer, select_activation

def calc_conv1d_output_size(input_size, kernel_size, padding, stride):
    return (input_size + 2 * padding - kernel_size) // stride + 1

def calc_conv_output_size(input_size, kernel_size, padding, stride):
    if isinstance(kernel_size, int):
        kernel_size = repeat(kernel_size, len(input_size))

    if isinstance(stride, int):
        stride = repeat(stride, len(input_size))

    if isinstance(padding, int):
        padding = repeat(padding, len(input_size))

    return tuple(
        calc_conv1d_output_size(sz, ks, p, s)
            for (sz, ks, p, s) in zip(input_size, kernel_size, padding, stride)
    )

class ResNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, rezero = False,
        kernel_size = 3, bottlneck_features = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if bottlneck_features is None:
            bottlneck_features = features

        self.block = nn.Sequential(
            nn.Conv2d(
                features, bottlneck_features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1,
            ),
            select_norm_layer(norm, bottlneck_features),
            select_activation(activ),

            nn.Conv2d(
                bottlneck_features, features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1
            ),
            select_norm_layer(norm, features),
        )

        self.block_out = select_activation(activ)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.block(x)
        z = x + self.re_alpha * y

        return self.block_out(z)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class ResNetBlockv2(nn.Module):

    def __init__(
        self, features, activ, norm, rezero = False,
        kernel_size = 3, bottlneck_features = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if bottlneck_features is None:
            bottlneck_features = features

        self.block = nn.Sequential(
            select_norm_layer(norm, bottlneck_features),
            select_activation(activ),

            nn.Conv2d(
                features, bottlneck_features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1,
            ),

            select_norm_layer(norm, features),
            select_activation(activ),

            nn.Conv2d(
                bottlneck_features, features,
                kernel_size = kernel_size,
                padding     = 'same',
                stride      = 1
            ),
        )

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.block(x)
        z = x + self.re_alpha * y

        return self.block_out(z)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class SpatialAttention(nn.Module):

    def __init__(
        self, features, norm, n_heads, rezero = False, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        self.norm     = select_norm_layer(norm, features)
        self.proj_in  = nn.Conv2d(features, 3 * features, kernel_size = 1)
        self.proj_out = nn.Conv2d(features, features,     kernel_size = 1)

        self._features = features
        self._n_heads  = n_heads

        assert self._features % self._n_heads == 0

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def _apply_attention(self, qkv):
        # qkv : (N, 3*C, H, W)
        (N, C3, H, W) = qkv.shape
        C = self._features

        # qkv : (N, 3*C, H*W)
        qkv = qkv.reshape(N, C3, H*W)

        # qkv : (N, heads, 3*C//heads, H*W)
        hqkv = qkv.reshape(N, self._n_heads, C3//self._n_heads, H*W)

        # hqkv : (N, heads, H*W, 3*C//heads)
        hqkv = hqkv.transpose(2, 3)

        # hq : (N, heads, H*W, C//heads)
        # hk : (N, heads, H*W, C//heads)
        # hv : (N, heads, H*W, C//heads)
        (hq, hk, hv) = torch.split(hqkv, C//self._n_heads, dim = 3)

        # ho : (N, heads, H*W, C//heads)
        ho = nn.functional.scaled_dot_product_attention(hq, hk, hv)

        # ho : (N, heads, C//heads, H*W)
        ho = ho.transpose(2, 3)

        # o : (N, C, H*W)
        o = ho.reshape(N, C, H*W)

        # (N, C, H, W)
        return o.reshape(N, C, H, W)

    def forward(self, x):
        # x : (N, C, H, W)

        # qkv : (N, 3 * C, H, W)
        qkv = self.proj_in(self.norm(x))
        o   = self._apply_attention(qkv)

        result = x + self.re_alpha * self.proj_out(o)

        return result

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class ResNetStem(nn.Module):

    def __init__(
        self, input_shape, features, norm = None, activ = None,
        kernel_size = 4, padding = 0, stride = 4
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                input_shape[0], features,
                kernel_size = kernel_size, padding = padding, stride = stride
            ),
            select_norm_layer(norm, features),
            select_activation(activ),
        )

        self._input_shape  = input_shape
        self._output_shape = (
            features,
            *calc_conv_output_size(
                input_shape[1:], kernel_size, padding, stride
            )
        )

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        return self.net(x)

class ResNetConv(nn.Module):

    def __init__(
        self, input_shape, features, kernel_size = 1, stride = 1, padding = 0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            input_shape[0], features,
            kernel_size = kernel_size, padding = padding, stride = stride,
        )

        self._input_shape  = input_shape
        self._output_shape = (
            features,
            *calc_conv_output_size(
                input_shape[1:], kernel_size, padding, stride
            )
        )

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        return self.conv(x)


class ResNetEncoder(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def _make_tv_resnet(self, block_spec, curr_shape):
        from .tv_models import TVResNetEncoder

        block      = TVResNetEncoder(curr_shape, **block_spec)
        curr_shape = block.calc_output_shape(curr_shape)

        return (block, curr_shape)

    def _make_stem_block(self, block_spec, curr_shape):
        warnings.warn(
            "ResNetEncoder block type 'stem' is deprecated: the name suggests"
            " Conv+Norm+Activ but this builder does not inject the encoder's"
            " norm/activ, so the block is a bare Conv2d unless the spec passes"
            " 'norm' and 'activ' explicitly. Use 'correct-stem' for a real"
            " stem (Conv+Norm+Activ) or 'conv' for an honest bare Conv2d.",
            DeprecationWarning,
            stacklevel = 2,
        )

        block = ResNetStem(curr_shape, **block_spec)
        curr_shape = block.output_shape

        return (block, curr_shape)

    def _make_correct_stem_block(self, block_spec, curr_shape):
        block = ResNetStem(
            curr_shape, norm = self._norm, activ = self._activ, **block_spec
        )
        curr_shape = block.output_shape

        return (block, curr_shape)

    def _make_conv_block(self, block_spec, curr_shape):
        block = ResNetConv(curr_shape, **block_spec)
        curr_shape = block.output_shape

        return (block, curr_shape)

    def _make_atten_block(self, block_spec, curr_shape):
        block = SpatialAttention(
            curr_shape[0], norm = self._norm, rezero = self._rezero,
            **block_spec
        )
        return (block, curr_shape)

    def _make_resample_block(self, block_spec, curr_shape):
        if isinstance(block_spec, (list, tuple)):
            size, kwargs = block_spec
        else:
            size   = block_spec
            kwargs = {}

        if isinstance(size, int):
            size = (size, size)

        features   = curr_shape[0]
        block      = nn.Upsample(size, **kwargs)
        curr_shape = (features, *size)

        return (block, curr_shape)

    def _make_resnet_block(self, block_spec, curr_shape):
        if isinstance(block_spec, (list, tuple)):
            n_blocks, kwargs = block_spec
        else:
            n_blocks = block_spec
            kwargs   = {}

        features = curr_shape[0]
        block    = nn.Sequential(
            *[ ResNetBlock(
                features,
                activ  = self._activ,
                norm   = self._norm,
                rezero = self._rezero,
                **kwargs
            )
            for _ in range(n_blocks) ]
        )

        return (block, curr_shape)

    def _make_resnet_block_v2(self, block_spec, curr_shape):
        if isinstance(block_spec, (list, tuple)):
            n_blocks, kwargs = block_spec
        else:
            n_blocks = block_spec
            kwargs   = {}

        features = curr_shape[0]
        block    = nn.Sequential(
            ResNetBlockv2(
                features,
                activ  = self._activ,
                norm   = self._norm,
                rezero = self._rezero,
                **kwargs
            ) for _ in range(n_blocks)
        )

        return (block, curr_shape)

    def __init__(
        self, input_shape, block_specs, activ, norm, rezero = True
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        super().__init__()

        curr_shape  = input_shape
        self.blocks = nn.ModuleList()

        self._activ  = activ
        self._norm   = norm
        self._rezero = rezero

        block_idx    = 0
        skip_indices = defaultdict(int)
        skip_shapes  = []

        for (block_type, block_spec) in block_specs:
            if block_type == 'stem':
                block, curr_shape \
                    = self._make_stem_block(block_spec, curr_shape)

            elif block_type == 'correct-stem':
                block, curr_shape \
                    = self._make_correct_stem_block(block_spec, curr_shape)

            elif block_type == 'conv':
                block, curr_shape \
                    = self._make_conv_block(block_spec, curr_shape)

            elif block_type == 'standard-resnet':
                # pylint: disable=import-outside-toplevel
                block, curr_shape \
                    = self._make_tv_resnet(block_spec, curr_shape)

            elif block_type == 'attention':
                block, curr_shape \
                    = self._make_atten_block(block_spec, curr_shape)

            elif block_type == 'resample':
                block, curr_shape \
                    = self._make_resample_block(block_spec, curr_shape)

            elif block_type == 'resnet':
                block, curr_shape \
                    = self._make_resnet_block(block_spec, curr_shape)

            elif block_type == 'resnet-v2':
                block, curr_shape \
                    = self._make_resnet_block_v2(block_spec, curr_shape)

            elif block_type == 'skip':
                skip_indices[block_idx] += 1
                skip_shapes.append(curr_shape)
                continue

            else:
                raise ValueError(f"Unknown block type: {block_type}")

            self.blocks.append(block)
            block_idx += 1

        self._input_shape  = input_shape
        self._output_shape = curr_shape
        self._skip_shapes  = skip_shapes
        self._skip_indices = skip_indices

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def skip_indices(self):
        return self._skip_indices

    @property
    def skip_shapes(self):
        return self._skip_shapes

    def forward(self, x, return_skips = False):
        if return_skips:
            skips = []

        y = x

        for idx, block in enumerate(self.blocks):
            if return_skips and (idx in self._skip_indices):
                skips += self._skip_indices[idx] * [ y, ]

            y = block(y)

        if return_skips:
            return (y, skips)

        return y

class ResNetFPN(nn.Module):

    def __init__(
        self, input_shape, block_specs, activ, norm, rezero = True
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._resnet = ResNetEncoder(
            input_shape, block_specs, activ, norm, rezero
        )

    @property
    def fpn_shapes(self):
        return self._resnet.skip_shapes + [ self._resnet.output_shape, ]

    def forward(self, x):
        y, skips = self._resnet(x, return_skips = True)

        result = skips
        result.append(y)

        return result

class PluggableResNetEncoder(ResNetEncoder):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, input_shape, block_specs, activ, norm, rezero = True
    ):
        # pylint: disable=too-many-arguments
        super(ResNetEncoder, self).__init__()

        curr_shape  = input_shape
        self.blocks = nn.ModuleList()

        self._activ  = activ
        self._norm   = norm
        self._rezero = rezero

        block_idx      = 0
        skip_indices   = set()
        skip_shapes    = []

        plugin_indices = set()
        plugin_args    = []

        for (block_type, block_spec) in block_specs:
            if block_type == 'stem':
                block, curr_shape \
                    = self._make_stem_block(block_spec, curr_shape)

            elif block_type == 'correct-stem':
                block, curr_shape \
                    = self._make_correct_stem_block(block_spec, curr_shape)

            elif block_type == 'conv':
                block, curr_shape \
                    = self._make_conv_block(block_spec, curr_shape)

            elif block_type == 'resample':
                block, curr_shape \
                    = self._make_resample_block(block_spec, curr_shape)

            elif block_type == 'resnet':
                block, curr_shape \
                    = self._make_resnet_block(block_spec, curr_shape)

            elif block_type == 'resnet-v2':
                block, curr_shape \
                    = self._make_resnet_block_v2(block_spec, curr_shape)

            elif block_type == 'skip':
                skip_indices.add(block_idx)
                skip_shapes.append(curr_shape)
                continue

            elif block_type == 'plugin':
                plugin_indices.add(block_idx)
                plugin_args.append(block_spec)
                continue

            else:
                raise ValueError(f"Unknown block type: {block_type}")

            self.blocks.append(block)
            block_idx += 1

        self._input_shape    = input_shape
        self._output_shape   = curr_shape
        self._skip_shapes    = skip_shapes
        self._skip_indices   = skip_indices
        self._plugin_indices = plugin_indices
        self._plugin_args    = plugin_args

    def forward(self, x, plugin, return_skips = False):
        # pylint: disable=arguments-renamed
        if return_skips:
            skips = []

        y = x
        plugin_idx = 0

        for idx, block in enumerate(self.blocks):
            if idx in self._plugin_indices:
                plugin_args  = self._plugin_args[plugin_idx]
                plugin_idx  += 1

                y = plugin(y, **plugin_args)

            if return_skips and (idx in self._skip_indices):
                skips.append(y)

            y = block(y)

        if return_skips:
            return (y, skips)

        return y

class PluggableResNetFPN(nn.Module):

    def __init__(
        self, input_shape, block_specs, activ, norm, rezero = True
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._resnet = PluggableResNetEncoder(
            input_shape, block_specs, activ, norm, rezero
        )

    @property
    def fpn_shapes(self):
        return self._resnet.skip_shapes + [ self._resnet.output_shape, ]

    def forward(self, x, plugin):
        y, skips = self._resnet(x, plugin, return_skips = True)

        result = skips
        result.append(y)

        return result

