import torch
import torch.nn.functional as F

from .funcs import match_shape
from .vlb import vlb_continuous

def l_simple(pred_eps, eps):
    return F.mse_loss(pred_eps, eps)

def l_vlb_continuous(
    x0, xt, t, pred_eps, pred_logvar, true_eps, vsched, sg_mean
):
    # pylint: disable=too-many-arguments
    tt = match_shape(t, xt)

    q_mu     = vsched.imu_eps(tt, xt, true_eps)
    q_var    = vsched.ivar(tt)
    q_logvar = torch.log(q_var)

    if sg_mean:
        p_mu = vsched.imu_eps(tt, xt, pred_eps.detach())
    else:
        p_mu = vsched.imu_eps(tt, xt, pred_eps)

    p_logvar = pred_logvar
    p_var    = torch.exp(p_logvar)

    return vlb_continuous(x0, tt, q_mu, q_logvar, q_var, p_mu, p_logvar, p_var)

