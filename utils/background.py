import numpy as np
from scipy.stats import gamma as gam

def sample_bg(data, chain, psf_stack, norm1_psf_stack, accept_bg, temp):
    """
    This function samples a new background intensity based on MCMC Algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        The input data.
    chain : dict
        The current chain values.
    psf_stack : numpy.ndarray
        The current PSF stack.
    norm1_psf_stack : numpy.ndarray
        The normalized PSF stack.
    accept_bg : int
        The number of accepted background intensities.
    temp : float
        The temperature.

    Returns
    -------
    bg : numpy.ndarray
        The new background intensity.
    psf_stack : numpy.ndarray
        The new PSF stack.
    accept_bg : int
        The number of accepted background intensities.
    log_prior : float
        The log of the prior probability.

    """
    bg = chain['Bg']
    i_val = chain['I']
    alpha_prop = 10000
    alpha = 1.1
    beta = 120
    t_bg = gam.rvs(a=alpha_prop, scale=bg/alpha_prop)

    # Calculate the new PSF_stack based on the new background intensity
    t_psf = i_val * norm1_psf_stack + t_bg[None, None, None, :]

    # Calculate the likelihood, prior, proposal, and posterior probabilities
    d_log_l = np.sum(data * (np.log(t_psf) - np.log(psf_stack)) - (t_psf - psf_stack)) / temp
    d_log_prior = np.sum(np.log(gam.pdf(t_bg, alpha, beta)) - np.log(gam.pdf(bg, alpha, beta)))
    d_log_prop = np.sum(np.log(gam.pdf(bg, alpha_prop, t_bg/alpha_prop)) - np.log(gam.pdf(t_bg, alpha_prop, bg/alpha_prop)))
    d_log_post = d_log_l + d_log_prior + d_log_prop

    # If the posterior probability is greater than a random number drawn from a uniform distribution between 0 and 1,
    # accept the new background intensity
    if d_log_post > np.log(np.random.rand()):
        bg = t_bg.copy()
        psf_stack = t_psf
        accept_bg += 1

    log_prior = np.sum(np.log(gam.pdf(t_bg, alpha, beta)))

    return bg, psf_stack, accept_bg, log_prior
