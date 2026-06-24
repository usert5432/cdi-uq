from cdi.bundled.leanbase.base.funcs  import extract_name_kwargs

from .param_cond_transformer import (
    ParamCondTransformer, ParamImageCondTrans
)

DM_DICT = {
    'param-cond-trans'       : ParamCondTransformer,
    'param-image-cond-trans' : ParamImageCondTrans,
}

def select_conditional_dm(dm, target_shape):
    name, kwargs = extract_name_kwargs(dm)

    if name not in DM_DICT:
        raise ValueError(f"Unknown encoder: '{name}'")

    return DM_DICT[name](
        target_shape = target_shape, **kwargs
    )

def construct_conditional_dm(dm, target_shape, device):
    model = select_conditional_dm(dm.model, target_shape)
    model = model.to(device)

    return model

