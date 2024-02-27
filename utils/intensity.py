import numpy as np
from scipy.stats import gamma as gam

def update_psf_stack(data, chain, psf_stack, norm_psf_stack, accept_count, temperature):
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
    norm_psf_stack : numpy.ndarray
        The normalized PSF stack.
    accept_count : int
        The number of accepted proposals.
    temperature : float
        The temperature parameter.

    Returns
    -------
    intensity : numpy.ndarray
        The updated intensity values.
    psf_stack : numpy.ndarray
        The updated PSF stack.
    accept_count : int
        The number of accepted proposals.
    log_prior : float
        The log-prior probability.

    """
    # Extract current parameter values
    background = chain['Bg']
    intensity = chain['I']

    # Set proposal distribution parameters
    alpha_proposal = 5000
    alpha = 8000
    beta = 0.5

    # Sample from the proposal distribution
    proposed_intensity = gam.rvs(a=alpha_proposal, scale=intensity/alpha_proposal)

    # Update the PSF stack using the new intensity
    proposed_psf = intensity * norm_psf_stack + np.expand_dims(background, axis=(1, 2, 3))

    # Compute the log-likelihood, log-prior, and log-proposal probabilities
    log_likelihood = np.sum(data * (np.log(proposed_psf) - np.log(psf_stack)) - (proposed_psf - psf_stack)) / temperature
    log_prior = np.log(gam.pdf(proposed_intensity, a=alpha, scale=beta)) - np.log(gam.pdf(intensity, a=alpha, scale=beta))
    log_proposal = np.log(gam.pdf(intensity, alpha_proposal, proposed_intensity/alpha_proposal)) - np.log(gam.pdf(proposed_intensity, alpha_proposal, intensity/alpha_proposal))
    log_posterior = log_likelihood + log_prior + log_proposal

    # Accept or reject the new proposal
    if log_posterior > np.log(np.random.rand()):
        intensity = proposed_intensity
        psf_stack = proposed_psf
        accept_count += 1

    # Compute and return the log-prior probability
    log_prior = np.log(gam.pdf(intensity, alpha, beta))
    return intensity, psf_stack, accept_count, log_prior
