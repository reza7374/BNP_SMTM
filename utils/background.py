import numpy as np
import scipy.stats as ss
from utils.functions import gampdf


def sample_background(data, chain, psf_stack, norm1_psf_stack, accept_bg, tp):
    """
    Sample the background intensity and update the PSF stack.

    Args:
        data (ndarray): Input data.
        chain (dict): Dictionary containing the chain variables.
        psf_stack (ndarray): PSF stack.
        norm1_psf_stack (ndarray): Normalized PSF stack.
        accept_bg (int): Counter for accepted background intensities.
        tp (float): Total probability.

    Returns:
        tuple: Updated background intensity, PSF stack, counter for accepted background intensities, and log prior.
    """

    background = chain['Bg']
    intensity = chain['I']
    alpha = 1.1
    beta = 120
    t_bg = ss.gamma.rvs(a=10000, scale=background/10000)
    t_psf = intensity * norm1_psf_stack + t_bg[:, np.newaxis, np.newaxis, np.newaxis]

    # Calculate the likelihood, prior, proposal, and posterior probabilities
    d_log_likelihood = np.sum(data * (np.log(t_psf) - np.log(psf_stack)) - (t_psf - psf_stack)) / tp
    d_log_prior = np.sum(np.log(gampdf(t_bg, alpha, beta)) - np.log(gampdf(background, alpha, beta)))
    d_log_proposal = np.sum(np.log(gampdf(background, 10000, t_bg/10000)) - np.log(gampdf(t_bg, 10000, background/10000)))
    d_log_posterior = d_log_likelihood + d_log_prior + d_log_proposal

    # Accept the new background intensity if the posterior probability is greater than a random number drawn from a uniform distribution between 0 and 1
    if d_log_posterior > np.log(np.random.rand()):
        background = t_bg
        psf_stack = t_psf
        accept_bg += 1

    log_prior = np.sum(np.log(gampdf(t_bg, alpha, beta)))

    return background, psf_stack, accept_bg, log_prior
