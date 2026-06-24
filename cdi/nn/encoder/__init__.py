from cdi.bundled.leanbase.base.funcs import extract_name_kwargs

from .resnet import ResNetEncoder, ResNetFPN
from .mlp import MLPConditionEncoder


ENCODERS = {
    'resnet'        : ResNetEncoder,
    'resnet-fpn'    : ResNetFPN,
    'mlp-condition' : MLPConditionEncoder,
}


def select_encoder(encoder, input_shape):
    name, kwargs = extract_name_kwargs(encoder)

    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: '{name}'")

    return ENCODERS[name](input_shape = input_shape, **kwargs)


def construct_encoder(encoder, input_shape, device):
    model = select_encoder(encoder.model, input_shape)
    model = model.to(device)

    return model
