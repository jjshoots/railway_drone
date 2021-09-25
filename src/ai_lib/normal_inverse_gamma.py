import torch
import torch.distributions as D

def NormalInvGamma(gamma, nu, alpha, beta):
    """
    Normal Inverse Gamma Distribution for evidential uncertainty learning:
        Deep Evidential Regression, Amini et. al.
            https://arxiv.org/pdf/1910.02600.pdf
            https://www.youtube.com/watch?v=toTcf7tZK8c
    """
    assert torch.all(nu > 0.), 'nu must be more than zero'
    assert torch.all(alpha > 1.), 'alpha must be more than one'
    assert torch.all(beta > 0.), 'beta must be more than zero'

    InvGamma = D.transformed_distribution.TransformedDistribution(
        D.Gamma(alpha, beta),
        D.transforms.PowerTransform(torch.tensor(-1.).to(gamma.device))
    )

    sigma_sq = InvGamma.rsample()
    mu = D.Normal(gamma, (1./nu) * sigma_sq).rsample()

    return D.Normal(mu, sigma_sq)

def uncertainty(alpha, beta):
    """
    calculates the aleotoric uncertainty of a distribution
    the value is effectively expectation of sigma square
    """
    assert torch.all(alpha > 1.), 'alpha must be more than one'
    assert torch.all(beta > 0.), 'beta must be more than zero'

    return beta / (alpha - 1)
