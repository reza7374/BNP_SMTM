import numpy as np
from scipy.stats import gamma as gam

def sample_intensity(data, chain, psf_stack, norm1_psf_stack, del_x, accept_i, tp):
    """
    This function updates the point spread function (PSF) stack using a Markov chain Monte Carlo (MCMC) algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        The data array.
    chain : dict
        The MCMC chain dictionary containing the current parameter values.
    psf_stack : numpy.ndarray
        The current PSF stack.
    norm1_psf_stack : numpy.ndarray
        The normalized PSF stack.
    del_x : float
        The PSF kernel size.
    accept_i : int
        The number of accepted proposals.
    tp : float
        The total number of points in the data.

    Returns
    -------
    i_val : numpy.ndarray
        The updated intensity values.
    psf_stack : numpy.ndarray
        The updated PSF stack.
    accept_i : int
        The number of accepted proposals.
    log_pr : float
        The log-prior probability.

    """
    # Extract current parameter values
    bg = chain['Bg']
    i_val = chain['I']

    # Set proposal distribution parameters
    alpha_prop = 5000
    alpha = 8000
    beta = 0.5

    # Sample from the proposal distribution
    t_i = gam.rvs(a=alpha_prop, scale=i_val/alpha_prop)

    # Update the PSF stack using the new intensity
    t_psf = i_val * norm1_psf_stack + np.expand_dims(bg, axis=(1, 2, 3))

    # Compute the log-likelihood, log-prior, and log-proposal probabilities
    d_log_l = np.sum(data * (np.log(t_psf) - np.log(psf_stack)) - (t_psf - psf_stack)) / tp
    d_log_prior = np.log(gam.pdf(t_i, a=alpha, scale=beta)) - np.log(gam.pdf(i_val, a=alpha, scale=beta))
    d_log_prop = np.log(gam.pdf(i_val, alpha_prop, t_i/alpha_prop)) - np.log(gam.pdf(t_i, alpha_prop, i_val/alpha_prop))
    d_log_post = d_log_l + d_log_prior + d_log_prop

    # Accept or reject the new proposal
    if d_log_post > np.log(np.random.rand()):
        i_val = t_i
        psf_stack = t_psf
        accept_i += 1

    # Compute and return the log-prior probability
    log_pr = np.log(gam.pdf(i_val, alpha, beta))
    return i_val, psf_stack, accept_i, log_pr