import torch

def normal_kl(mu1, mu2, logvar1, logvar2, var1 = None, var2 = None):
    # pylint: disable=too-many-arguments

    # KL(p1, p2) = 1/2 * (
    #   log(var2 / var1)
    #   + var1 / var2 - 1
    #   + (mu1 - mu2)^2 / var2
    # )

    if var1 is None:
        var1 = torch.exp(logvar1)

    if var2 is None:
        var2 = torch.exp(logvar2)

    p1 = logvar2 - logvar1
    p2 = var1 / var2 - 1
    p3 = (mu1 - mu2)**2 / var2

    return 1/2 * (p1 + p2 + p3)

def vlb_t(q_mu, q_logvar, q_var, p_mu, p_logvar, p_var):
    # pylint: disable=too-many-arguments
    # KL(q(x[t-1] | x[t], x0) || p(x[t-1] | x[t]))
    return normal_kl(q_mu, p_mu, q_logvar, p_logvar, q_var, p_var)

def vlb_1_continuous(x0, p_mu, p_var):
    # - log p(x[0] | x[1])
    d = torch.distributions.normal.Normal(p_mu, torch.sqrt(p_var))
    return -d.log_prob(x0)

def vlb_continuous(x0, t, q_mu, q_logvar, q_var, p_mu, p_logvar, p_var):
    # pylint: disable=too-many-arguments

    lt = vlb_t(q_mu, q_logvar, q_var, p_mu, p_logvar, p_var)
    l1 = vlb_1_continuous(x0, p_mu, p_var)

    return torch.where((t > 1), lt, l1).mean()


