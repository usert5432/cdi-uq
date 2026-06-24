import torch

from .dp     import DiffusionProcess
from .normal import (
    CondNorm, convolve_inc_pdfs, x_from_eps, subsample_inc_pdfs
)

from .subsample import generate_subsampling
from .vsched    import generate_variance_schedule

class DDIM(DiffusionProcess):

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

    def get_fwd_jump_p(self, t):
        return self._fwd_jump_ps[t]

    def get_bkw_step_p(self, t, x0):
        #
        # Constraints to match forward process
        #
        # var   = p_prev.var - scale**2 * p_curr.var
        # bias  = (
        #     (p_prev.scale - scale * p_curr.scale) * x0
        #   + (p_prev.bias  - scale * p_curr.bias)
        # )
        #
        # To make sure that var == 0, set
        #
        # scale = (p_prev.var / p_curr.var).sqrt()
        #

        p_prev = self._fwd_jump_ps[t-1].match_shape(x0)
        p_curr = self._fwd_jump_ps[t].match_shape(x0)

        scale_sq = (p_prev.var / p_curr.var)

        scale = scale_sq.sqrt()
        var   = torch.zeros_like(scale)
        bias  = (
              (p_prev.scale - scale * p_curr.scale) * x0
            + (p_prev.bias  - scale * p_curr.bias)
        )

        return CondNorm(scale, bias, var)

    def forward_jump(self, t, x0, eps = None):
        # pylint: disable=arguments-differ
        p = self._fwd_jump_ps[t].match_shape(x0)

        if eps is None:
            eps = self._generate_noise(x0)

        result = p.scale * x0 + p.bias + p.var.sqrt() * eps

        return (result, eps)

    def backward_step_given_x0(self, t, x, x0):
        # pylint: disable=arguments-differ
        p = self.get_bkw_step_p(t, x0)

        result = p.scale * x + p.bias

        return result

    def backward_step_given_eps(self, t, x, eps, **kwargs):
        p_jump = self._fwd_jump_ps[t].match_shape(x)
        x0     = x_from_eps(p_jump, x, eps)

        return self.backward_step_given_x0(t, x, x0, **kwargs)

    def subsample(self, subsample):
        # pylint: disable=arguments-differ
        subspace = generate_subsampling(subsample, len(self))

        vsched   = subsample_inc_pdfs(self._fwd_step_ps, subspace)
        time_map = self.map_time(subspace.to(self._device))

        return DDIM(vsched, self._device, time_map = time_map, prg = self._prg)

    def marginal_variance(self):
        return self._fwd_jump_ps.var[-1]

