from torch import nn
from ..encoder.resnet import ResNetEncoder

class ResNetRegressor(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, input_shape, output_shape, block_specs, activ, norm,
        rezero = False
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        super().__init__()

        assert len(output_shape) == 1

        self.encoder \
            = ResNetEncoder(input_shape, block_specs, activ, norm, rezero)

        encoder_out_features = self.encoder.output_shape[0]

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(encoder_out_features, output_shape[0])
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.output(z)

