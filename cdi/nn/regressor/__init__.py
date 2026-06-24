from cdi.bundled.leanbase.base.funcs  import extract_name_kwargs
from .mlp import MLPRegressor
from .resnet import ResNetRegressor
from .tv_models import TVResNetRegressor, TVEfficientNetRegressor

REGRESSORS = {
    'mlp'        : MLPRegressor,
    'resnet'     : ResNetRegressor,
    'tv-resnet'  : TVResNetRegressor,
    'tv-efficientnet'  : TVEfficientNetRegressor,
}

def select_regressor(regressor, input_shape, output_shape):
    name, kwargs = extract_name_kwargs(regressor)

    if name not in REGRESSORS:
        raise ValueError(f"Unknown regressor: '{name}'")

    return REGRESSORS[name](
        input_shape = input_shape, output_shape = output_shape, **kwargs
    )

def construct_regressor(regressor, input_shape, output_shape, device):
    model = select_regressor(regressor.model, input_shape, output_shape)
    model = model.to(device)

    return model
