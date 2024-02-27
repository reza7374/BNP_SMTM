import numpy as np
from utils.functions import bernoulli_pdf
from utils.psf import calculate_psf

def sample_load(data, chain, psf_stack, tmp_phase, del_x, defocus_k, mask, bnp, sub_pixel_zeros, start_ind, end_ind, sub_pixel, x_offset_phase, y_offset_phase, tp):
    """
    Function to sample loads and update PSF stack based on Metropolis-Hastings algorithm.

    Args:
        data (ndarray): Observed data.
        chain (dict): Dictionary containing chain variables.
        psf_stack (ndarray): Stack of PSFs.
        tmp_phase (ndarray): Temporary phase values.
        del_x (ndarray): Array of x, y, and z offsets.
        defocus_k (float): Defocus parameter.
        mask (ndarray): Mask for PSF calculation.
        bnp (dict): Dictionary containing BNP variables.
        sub_pixel_zeros (ndarray): Array of sub-pixel zeros.
        start_ind (int): Start index for sub-pixel calculation.
        end_ind (int): End index for sub-pixel calculation.
        sub_pixel (float): Sub-pixel value.
        x_offset_phase (float): X offset for phase calculation.
        y_offset_phase (float): Y offset for phase calculation.
        tp (float): Time parameter.

    Returns:
        tuple: Tuple containing updated loads and PSF stack.
    """

    phase = chain['Phase']  # Get the phase values from the chain dictionary
    mag = chain['Mag']  # Get the magnitude values from the chain dictionary
    bg = chain['Bg']  # Get the background values from the chain dictionary
    intensity = chain['I']  # Get the intensity values from the chain dictionary
    x_coord = chain['X']  # Get the x-coordinate values from the chain dictionary
    y_coord = chain['Y']  # Get the y-coordinate values from the chain dictionary
    z_coord = chain['Z']  # Get the z-coordinate values from the chain dictionary
    loads = chain['loads']  # Get the loads from the chain dictionary
    
    num_loads = np.size(loads)  # Get the number of loads
    temp_loads = np.copy(loads)  # Create a temporary copy of the loads
    
    gm = bnp['gm']  # Get the gamma value from the bnp dictionary
    
    random_indices = np.random.choice(num_loads, 2, replace=False)  # Randomly choose 2 load indices
    load_pairs = [(x, y) for x in [0, 1] for y in [0, 1]]  # Create all possible load pairs
    
    for it in range(4):
        load_index_1, load_index_2 = load_pairs[it]  # Get the load indices for the current iteration
        
        temp_loads[random_indices[0]] = load_index_1  # Update the temporary loads with load_index_1
        temp_loads[random_indices[1]] = load_index_2  # Update the temporary loads with load_index_2
        
        temp_psf = calculate_psf(mag, tmp_phase, bg, intensity, temp_loads, defocus_k, z_coord+del_x[:, 2], mask, sub_pixel_zeros, start_ind, end_ind, sub_pixel, x_coord+del_x[:, 0], y_coord+del_x[:, 1], x_offset_phase, y_offset_phase)  # Calculate the temporary PSF
        
        log_likelihood = np.sum(data*(np.log(temp_psf)-np.log(psf_stack))-(temp_psf-psf_stack))  # Calculate the log likelihood
        log_prior = np.sum(np.log(bernoulli_pdf(temp_loads, num_loads, gm)) - np.log(bernoulli_pdf(loads, num_loads, gm)))  # Calculate the log prior
        log_posterior = log_likelihood + log_prior  # Calculate the log posterior
        
        if log_posterior > np.log(np.random.rand()):  # Accept the new loads with a certain probability
            loads[random_indices[0]] = load_index_1  # Update the loads with load_index_1
            loads[random_indices[1]] = load_index_2  # Update the loads with load_index_2
            psf_stack = temp_psf.copy()  # Update the PSF stack

    return loads, psf_stack  # Return the updated loads and PSF stack
