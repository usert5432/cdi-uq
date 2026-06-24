import torch
from .funcs import match_shape

#
# p(y | x) = N(y; scale * x + bias; var)
#

class CondNorm:

    __slots__ = [
        'scale', 'bias', 'var'
    ]

    def __init__(self, scale, bias, var):
        self.scale = scale
        self.bias  = bias
        self.var   = var

    def __getitem__(self, index):
        scale = self.scale[index]
        bias  = self.bias[index]
        var   = self.var[index]

        return CondNorm(scale, bias, var)

    def match_shape(self, target):
        if self.scale.ndim != 1:
            return self

        scale = match_shape(self.scale, target)
        bias  = match_shape(self.bias, target)
        var   = match_shape(self.var, target)

        return CondNorm(scale, bias, var)

    def to(self, device):
        scale = self.scale.to(device)
        bias  = self.bias.to(device)
        var   = self.var.to(device)

        return CondNorm(scale, bias, var)

    def __len__(self):
        return len(self.scale)

    def __repr__(self):
        result  = 'CondNorm({\n'
        result += f'  scale = {self.scale}\n'
        result += f'  bias  = {self.bias}\n'
        result += f'  var   = {self.var}\n'
        result += '}'

        return result

def cat_cond_norms(cond_norms):
    scale = torch.cat([x.scale for x in cond_norms])
    bias  = torch.cat([x.bias  for x in cond_norms])
    var   = torch.cat([x.var   for x in cond_norms])

    return CondNorm(scale, bias, var)

def convolve_cond_norm_pdfs(p_zy : CondNorm, p_yx : CondNorm) -> CondNorm:
    # p(z | x) = \int dy p_p_zy(z | y) * p_p_yx(y | x)

    scale = p_zy.scale * p_yx.scale
    bias  = p_zy.bias + p_zy.scale    * p_yx.bias
    var   = p_zy.var  + p_zy.scale**2 * p_yx.var

    return CondNorm(scale, bias, var)

def invert_cond_norm_pdfs(p_zy : CondNorm, p_yx : CondNorm, x) -> CondNorm:
    # p(y | z, x) = p(z | y) * p(y | x) / p(z | x)

    norm  = (p_zy.var + p_zy.scale**2 * p_yx.var)

    scale = p_zy.scale * p_yx.var / norm

    bias  = (
          p_yx.scale * x * p_zy.var
        + p_yx.bias * p_zy.var
        - p_zy.scale * p_zy.bias * p_yx.var
    ) / norm

    var   = p_zy.var * p_yx.var / norm

    return CondNorm(scale, bias, var)

def x_from_eps(p_yx : CondNorm, y, eps):
    # y = scale * x + bias + sqrt(var) * eps
    #
    # x = 1 / scale * (y - bias - sqrt(var) * eps)

    return 1 / p_yx.scale * (y - p_yx.bias - p_yx.var.sqrt() * eps)

def convolve_norm_scale_arr(scale):
    # p(z | x) = \int dy p_curr(z | y) * p_prev(y | x)
    #
    # scale = curr.scale * prev_cumul.scale

    return torch.cumprod(scale, dim = 0)

def convolve_norm_bias_arr(scale, bias):
    # p(z | x) = \int dy p_curr(z | y) * p_prev(y | x)
    #
    # bias  = curr.bias + curr.scale * prev_cumul.bias

    result    = torch.zeros_like(bias)
    result[0] = bias[0]

    for i in range(1, len(result)):
        result[i] = bias[i] + scale[i] * result[i-1]

    return result

def convolve_norm_var_arr(scale, var):
    # p(z | x) = \int dy p_curr(z | y) * p_prev(y | x)
    #
    # var = curr.var + curr.scale**2 * prev_cumul.var

    result    = torch.zeros_like(var)
    result[0] = var[0]

    for i in range(1, len(result)):
        result[i] = var[i] + scale[i]**2 * result[i-1]

    return result

def convolve_inc_pdfs(p_inc_list : CondNorm) -> CondNorm:
    result_scale = convolve_norm_scale_arr(p_inc_list.scale)
    result_bias  = convolve_norm_bias_arr(p_inc_list.scale, p_inc_list.bias)
    result_var   = convolve_norm_var_arr(p_inc_list.scale, p_inc_list.var)

    return CondNorm(result_scale, result_bias, result_var)

def subsample_inc_pdfs(
    step_ps : CondNorm, subspace : torch.Tensor
) -> CondNorm:

    t_prev = 0
    result = [ ]

    for t in subspace:
        t_curr = t.cpu().item()

        jump_ps = convolve_inc_pdfs(step_ps[t_prev:t_curr+1])
        p_jump  = jump_ps[-1:]

        result.append(p_jump)

        t_prev = t_curr+1

    return cat_cond_norms(result)

