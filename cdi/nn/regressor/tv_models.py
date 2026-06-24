import copy
import logging

from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.models import (
    resnet18, resnet34, resnet50,
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
)

from torchvision.models.resnet import ResNet as TVResNet
from torchvision.models.resnet import BasicBlock as TVResNetBasicBlock

def resnet10(**kwargs):
    return TVResNet(TVResNetBasicBlock, [1, 1, 1, 1], **kwargs)

RESNET_MODELS = {
    'resnet-10' : resnet10,
    'resnet-18' : resnet18,
    'resnet-34' : resnet34,
    'resnet-50' : resnet50,
}

EFFICIENT_NET_MODELS = {
    'efficientnet-b0' : efficientnet_b0,
    'efficientnet-b1' : efficientnet_b1,
    'efficientnet-b2' : efficientnet_b2,
    'efficientnet-b3' : efficientnet_b3,
}

LOGGER = logging.getLogger('cdi.nn.regressors')

def fix_bn_momentum(model, bn_momentum):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            LOGGER.debug(
                "Adjusting BatchNorm2d momentum from '%f' to '%f'",
                m.momentum, bn_momentum
            )
            m.momentum = bn_momentum

def clone_conv_with_new_in_channels(conv_layer, new_in_channels):
    if not isinstance(conv_layer, nn.Conv2d):
        raise TypeError(f"Expected nn.Conv2d, got {type(conv_layer)}")

    return nn.Conv2d(
        in_channels  = new_in_channels,
        out_channels = conv_layer.out_channels,
        kernel_size  = conv_layer.kernel_size,
        stride       = conv_layer.stride,
        padding      = conv_layer.padding,
        dilation     = conv_layer.dilation,
        groups       = conv_layer.groups,
        bias         = conv_layer.bias is not None,
        padding_mode = conv_layer.padding_mode,
        device       = conv_layer.weight.device,
        dtype        = conv_layer.weight.dtype
    )

def clone_convnormact_with_new_in_channels(conv_layer, new_in_channels):
    if not isinstance(conv_layer, Conv2dNormActivation):
        raise TypeError(
            f"Expected Conv2dNormActivation, got {type(conv_layer)}"
        )

    result = copy.deepcopy(conv_layer)
    result[0] = clone_conv_with_new_in_channels(result[0], new_in_channels)

    return result

class TVResNetRegressor(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, input_shape, output_shape, model_type, bn_momentum = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        assert len(output_shape) == 1

        if model_type not in RESNET_MODELS:
            raise ValueError(
                f"Unknown torchvision resnet model type: '{model_type}'"
            )

        self.model = RESNET_MODELS[model_type](
            num_classes = output_shape[0], **kwargs
        )

        if input_shape[0] != 3:
            self.model.conv1 = clone_conv_with_new_in_channels(
                self.model.conv1, input_shape[0]
            )

        if bn_momentum is not None:
            fix_bn_momentum(self.model, bn_momentum)

    def forward(self, x):
        return self.model(x)

class TVEfficientNetRegressor(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, input_shape, output_shape, model_type, bn_momentum = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        assert len(output_shape) == 1

        if model_type not in EFFICIENT_NET_MODELS:
            raise ValueError(
                f"Unknown torchvision efficient net model type: '{model_type}'"
            )

        self.model = EFFICIENT_NET_MODELS[model_type](
            num_classes = output_shape[0], **kwargs
        )

        if input_shape[0] != 3:
            self.model.features[0] = clone_convnormact_with_new_in_channels(
                self.model.features[0], input_shape[0]
            )

        if bn_momentum is not None:
            fix_bn_momentum(self.model, bn_momentum)

    def forward(self, x):
        return self.model(x)

