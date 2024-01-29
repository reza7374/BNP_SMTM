import numpy as np
from functions import inv_gamma_pdf


def sample_diff(chain, del_x, dt):
    """
    Calculates the diffusion coefficient and its log-prior probability given a chain, del_x, and dt.

    Args:
        chain (dict): Dictionary containing information about the chain, including 'loads', 'X', 'Y', and 'Z'.
        del_x (numpy.ndarray): Numpy array of shape (n,3) representing the displacement of the particle in 3D space over time.
        dt (float): Time interval.

    Returns:
        D (float): Diffusion coefficient calculated based on the provided information.
        l_prior_D (float): Log-prior probability of D calculated based on the provided information.
    """
    
    # Get loads, alpha_prior, and beta_prior from chain
    loads = chain['loads']
    alpha_prior = 13
    beta_prior = 1200
    
    # Check if any particle is active
    if np.sum(loads) > 0:
        
        # Get X, Y, and Z from chain
        x = chain['X']; y = chain['Y']; z = chain['Z']
        
        # Calculate alpha using R, NN, and size of X
        R = np.size(del_x, axis=0)
        NN = np.sum(loads)
        alpha = (3/2) * R * NN * (np.size(x, axis=0) - 1) + alpha_prior
        
        # Calculate beta using R, X, Y, Z, and dt
        beta = beta_prior
        for nn in [nn for nn in range(np.size(z, axis=1)) if loads[nn] != 0]:
            beta += (R * np.sum((x[1:, nn] - x[:-1, nn])**2 + (y[1:, nn] - y[:-1, nn])**2 + (z[1:, nn] - z[:-1, nn])**2))/(4 * dt)
        
        # Generate D using a gamma distribution with alpha and beta values
        D = 1 / np.random.gamma(shape=alpha, scale=1/beta)
        
        # Calculate log-prior probability of D
        l_prior_D = np.log(inv_gamma_pdf(D, alpha_prior, beta_prior))
    
    # If there is no active particle, use alpha_prior and beta_prior to calculate D
    else:
        D = 1 / np.random.gamma(shape=alpha_prior, scale=1/beta_prior)
        l_prior_D = np.log(inv_gamma_pdf(D, alpha_prior, beta_prior))
    
    # Return D and its log-prior probability l_prior_D
    return D, l_prior_D

