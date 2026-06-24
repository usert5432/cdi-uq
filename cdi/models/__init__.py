from cdi.bundled.leanbase.base.funcs import extract_name_kwargs

from .cdi import CDI
from .regression import Regressor

MODELS_DICT = {
    'cdi'             : CDI,
    'regressor'       : Regressor,
}

def select_model(model, **kwargs):
    name, model_kwargs = extract_name_kwargs(model)

    if name not in MODELS_DICT:
        raise ValueError(f"Unknown model: '{name}'")

    return MODELS_DICT[name](**kwargs, **model_kwargs)

def construct_model(config, device, dtype, init_train, savedir):
    model = select_model(
        config.model,
        config     = config,
        device     = device,
        dtype      = dtype,
        init_train = init_train,
        savedir    = savedir,
    )

    return model
