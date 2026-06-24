import torch
from cdi.bundled.leanbase.base.funcs import extract_name_kwargs

#
# Forward Diffusion:
#
# q(x(t) | x(t-1)) = N(x(t); scale(t | t-1) * x(t-1), var(t | t-1))
#
# q(x(t) | x0) = N(x(t), scale(t) * x0, var(t))
#
# where
#   scale(t) = scale(t | t-1) * scale(t-1)
#   var(t)   = var(t | t-1) + scale(t | t-1)^2 * var(t-1)
#
#
# Inverse:
#
# q(x(t-1) | x(t), x0) = N(x(t-1); imu(x(t), x0), ivar(t))
#
# where
#   imu(x(t), x0) = (
#       scale(t | t-1) * var(t-1) / var(t) * x(t)
#     + scale(t-1) * var(t | t-1) / var(t) * x0
#   )
#   ivar(t) = var(t | t-1) * var(t-1) / var(t)
#
# Inverse Epsilon:
#
#   imu(x(t), x0) = (
#       1 / scale(t | t-1) * x(t)
#     - var(t | t-1) / (sqrt(var(t)) * scale(t | t-1)) * eps
#   )
#

def calc_cumul_scale(scale_step):
    # scale(t) = scale(t | t-1) * scale(t-1)
    return torch.cumprod(scale_step, dim = 0)

def calc_cumul_var(scale_step, var_step):
    # var(t) = var(t | t-1) + scale(t | t-1)^2 * var(t-1)
    result = []
    result.append(var_step[0])

    for t in range(1, len(var_step)):
        var = var_step[t] + scale_step[t]**2 * result[-1]
        result.append(var)

    return torch.Tensor(result) \
        .to(device = scale_step.device, dtype = var_step.dtype)

class VarianceSchedule:

    def __init__(self, scale_step, var_step, device):
        self._scale_step = torch.Tensor(scale_step) .to(device)
        self._var_step   = torch.Tensor(var_step).to(device)

        self._scale = calc_cumul_scale (self._scale_step)
        self._var   = calc_cumul_var(self._scale_step, self._var_step)

    def __len__(self):
        return len(self._scale_step) - 1

    @property
    def T(self):
        return len(self)

    @property
    def scale_step(self):
        return self._scale_step

    @property
    def var_step(self):
        return self._var_step

    @property
    def scale(self):
        return self._scale

    @property
    def var(self):
        return self._var

    def ivar(self, t):
        # ivar(t) = var(t | t-1) * var(t-1) / var(t)
        return self.var_step[t] * self.var[t-1] / self.var[t]

    def imu(self, t, xt, x0):
        #   imu(x(t), x0) = (
        #       scale(t | t-1) * var(t-1) / var(t) * x(t)
        #     + scale(t-1) * var(t | t-1) / var(t) * x0
        #   )

        pt = self.scale_step[t] * self.var[t-1] / self.var[t] * xt
        p0 = self.scale[t-1] * self.var_step[t] / self.var[t] * x0

        return pt + p0

    def imu_eps(self, t, xt, eps):
        #   imu(x(t), x0) = (
        #       1 / scale(t | t-1) * x(t)
        #     - var(t | t-1) / (sqrt(var(t)) * scale(t | t-1)) * eps
        #   )

        pt = 1 / self.scale_step[t] * xt
        pe = self.var_step[t] / (self.var[t].sqrt() * self.scale_step[t]) * eps

        return pt - pe

def generate_linear_schedule(T = 1000, beta1 = 1e-4, betaT = 0.02):
    var_step = torch.linspace(beta1, betaT, T)
    pad      = torch.zeros((1,), dtype = var_step.dtype)

    var_step   = torch.cat((pad, var_step), dim = 0)
    scale_step = (1 - var_step).sqrt()

    return (scale_step, var_step)

def generate_cosine_schedule(T = 1000, s = 0.008, clip = 0.999):
    def fn(t):
        return torch.cos((t / T + s) / (1 + s) * torch.pi / 2).square()

    t = torch.arange(0, T+1)

    var_step = (1 - fn(t[1:]) / fn(t[:-1]))
    var_step = torch.clamp(var_step, 0, clip)
    pad      = torch.zeros((1,), dtype = var_step.dtype)

    var_step   = torch.cat((pad, var_step), dim = 0)
    scale_step = (1 - var_step).sqrt()

    return (scale_step, var_step)

def generate_variance_schedule(vsched, device):
    name, kwargs = extract_name_kwargs(vsched)

    if name == 'linear':
        scale_step, var_step = generate_linear_schedule(**kwargs)
        return VarianceSchedule(scale_step, var_step, device)

    if name == 'cosine':
        scale_step, var_step = generate_cosine_schedule(**kwargs)
        return VarianceSchedule(scale_step, var_step, device)

    raise ValueError(f"Unknown noise sched: {name}")

