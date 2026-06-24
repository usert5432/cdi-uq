import torch

from .dp     import DiffusionProcess
from .normal import (
    convolve_inc_pdfs, x_from_eps, invert_cond_norm_pdfs, subsample_inc_pdfs
)

from .subsample import generate_subsampling
from .vsched    import generate_variance_schedule
from .funcs     import match_shape

class DDPM(DiffusionProcess):

    def __init__(
        self, vsched, device, seed = None, time_map = None, prg = None
    ):
        # pylint: disable=too-many-arguments
        self._fwd_step_ps = generate_variance_schedule(vsched).to(device)
        self._fwd_jump_ps = convolve_inc_pdfs(self._fwd_step_ps)

        super().__init__(
            n        = len(self._fwd_step_ps)-1,
            device   = device,
            seed     = seed,
            time_map = time_map,
            prg      = prg,
        )

    def _generate_noise(self, x):
        return torch.randn(
            size = x.shape, generator = self._prg, device = x.device
        )

    def get_fwd_step_p(self, t):
        return self._fwd_step_ps[t]

    def get_fwd_jump_p(self, t):
        return self._fwd_jump_ps[t]

    def get_bkw_step_p(self, t, x0):
        #
        # p(x[t-1] | x[t], x0)
        #   = p(x[t] | x[t-1]) * p(x[t-1] | x0) / p(x[t] | x0)
        #
        # Product of (p_step[t] * p_jump[t-1])
        #

        p_step = self._fwd_step_ps[t].match_shape(x0)
        p_jump = self._fwd_jump_ps[t-1].match_shape(x0)

        p = invert_cond_norm_pdfs(p_step, p_jump, x0)

        return p

    def forward_step(self, t, x_prev, eps = None):
        # pylint: disable=arguments-differ

        p = self._fwd_step_ps[t].match_shape(x_prev)

        if eps is None:
            eps = self._generate_noise(x_prev)

        result = p.scale * x_prev + p.bias + p.var.sqrt() * eps

        return (result, eps)

    def forward_jump(self, t, x0, eps = None):
        # pylint: disable=arguments-differ

        p = self._fwd_jump_ps[t].match_shape(x0)

        if eps is None:
            eps = self._generate_noise(x0)

        result = p.scale * x0 + p.bias + p.var.sqrt() * eps

        return (result, eps)

    def backward_jump_given_eps(self, t, x, eps, **kwargs):
        p = self._fwd_jump_ps[t].match_shape(x)

        # x  = scale * x0 + bias + var.sqrt() * eps
        # x0 = (x - bias - var.sqrt() * esp) / scale

        result = (x - p.bias - p.var.sqrt() * eps) / p.scale

        return result

    def backward_step_given_x0(self, t, x, x0, var = None, bkw_eps = None):
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-arguments

        # p(x[t-1] | x[t], x0)
        #   = p(x[t] | x[t-1]) * p(x[t-1] | x0) / p(x[t] | x0)
        #
        # Product of (p_step[t] * p_jump[t-1])
        #

        p = self.get_bkw_step_p(t, x0)

        if var == 'fwd':
            p_step = self._fwd_step_ps[t-1].match_shape(x0)
            p.var  = p_step.var
        elif var is not None:
            p.var = var

        if bkw_eps is None:
            bkw_eps = self._generate_noise(x)
            bkw_eps = bkw_eps * (match_shape(self.map_time(t), x) > 1)

        result = p.scale * x + p.bias + p.var.sqrt() * bkw_eps

        return result

    def backward_step_given_eps(self, t, x, eps, **kwargs):
        p_jump = self._fwd_jump_ps[t].match_shape(x)
        x0     = x_from_eps(p_jump, x, eps)

        return self.backward_step_given_x0(t, x, x0, **kwargs)

    def subsample(self, subsample):
        subspace = generate_subsampling(subsample, len(self))

        vsched   = subsample_inc_pdfs(self._fwd_step_ps, subspace)
        time_map = self.map_time(subspace.to(self._device))

        return DDPM(vsched, self._device, time_map = time_map, prg = self._prg)

    def marginal_variance(self):
        return self._fwd_jump_ps.var[-1]

