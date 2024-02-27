import numpy as np
from functions import inv_gamma_pdf


def sample_diffusion_coefficient(chain, displacement, time_interval):
    """
    Calculates the diffusion coefficient and its log-prior probability given a chain, displacement, and time interval.

    Args:
        chain (dict): Dictionary containing information about the chain, including 'loads', 'x', 'y', and 'z'.
        displacement (numpy.ndarray): Numpy array of shape (n,3) representing the displacement of the particle in 3D space over time.
        time_interval (float): Time interval.

    Returns:
        diffusion_coefficient (float): Diffusion coefficient calculated based on the provided information.
        log_prior_probability (float): Log-prior probability of the diffusion coefficient calculated based on the provided information.
    """
    
    # Get loads, alpha_prior, and beta_prior from chain
    loads = chain['loads']
    alpha_prior = 13
    beta_prior = 1200
    
    # Check if any particle is active
    if np.sum(loads) > 0:
        # Get x, y, and z from chain
        x, y, z = chain['X'], chain['Y'], chain['Z']
        
        # Calculate alpha using number of displacements, number of active particles, and size of x
        num_displacements = displacement.shape[0]
        num_active_particles = np.sum(loads)
        alpha = (3/2) * num_displacements * num_active_particles * (x.shape[0] - 1) + alpha_prior
        
        # Calculate beta using number of displacements, x, y, z, and time interval
        squared_displacements = np.sum((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2)
        beta = beta_prior + (num_displacements * squared_displacements) / (4 * time_interval)
        
        # Generate diffusion coefficient using a gamma distribution with alpha and beta values
        diffusion_coefficient = 1 / np.random.gamma(shape=alpha, scale=1/beta)
        
        # Calculate log-prior probability of the diffusion coefficient
        log_prior_probability = np.log(inv_gamma_pdf(diffusion_coefficient, alpha_prior, beta_prior))
    
    else:
        # If there is no active particle, use alpha_prior and beta_prior to calculate the diffusion coefficient
        diffusion_coefficient = 1 / np.random.gamma(shape=alpha_prior, scale=1/beta_prior)
        log_prior_probability = np.log(inv_gamma_pdf(diffusion_coefficient, alpha_prior, beta_prior))
    
    # Return diffusion coefficient and its log-prior probability
    return diffusion_coefficient, log_prior_probability
