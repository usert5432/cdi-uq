import torch

from .funcs import match_shape

def diffusion_forward(t, x0, vsched, generator):
    # xt : (N, ...)
    # t  : (N,)

    # eps : (N, ...)
    eps = torch.randn(
        size = x0.shape, generator = generator, device = x0.device
    )

    result  = match_shape(vsched.scale[t], x0) * x0
    result += match_shape(vsched.var[t].sqrt(), eps) * eps

    return (result, eps)

def diffusion_backward_step(t, xt, eps, vsched, generator, var):
    # pylint: disable=too-many-arguments
    # xt  : (N, ...)
    # eps : (N, ...)
    # t   : (1,)

    z = torch.randn(
        size = xt.shape, generator = generator, device = xt.device
    )

    z = z * (t > 1)

    if var == 'fwd':
        var = vsched.var_step[t]
    elif var == 'bkw':
        var = vsched.ivar(t)

    return vsched.imu_eps(t, xt, eps) + var.sqrt() * z

