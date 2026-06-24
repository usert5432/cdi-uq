import logging

from torch import nn
from torchvision.models import (resnet18, resnet34, resnet50)

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

RESNET_OUTPUT_CHANNELS = {
    'resnet-10' : 512,
    'resnet-18' : 512,
    'resnet-34' : 512,
    'resnet-50' : 2048,
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

class TVResNetEncoder(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, input_shape, model_type, bn_momentum = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__()


        if model_type not in RESNET_MODELS:
            raise ValueError(
                f"Unknown torchvision resnet model type: '{model_type}'"
            )

        resnet = RESNET_MODELS[model_type](**kwargs)

        if input_shape[0] != 3:
            resnet.conv1 = clone_conv_with_new_in_channels(
                resnet.conv1, input_shape[0]
            )

        if bn_momentum is not None:
            fix_bn_momentum(resnet, bn_momentum)

        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self._output_channels = RESNET_OUTPUT_CHANNELS[model_type]

    def calc_output_shape(self, input_shape):
        return (
            self._output_channels, input_shape[1] // 32, input_shape[2] // 32
        )

    def forward(self, x):
        return self.model(x)

