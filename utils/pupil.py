import numpy as np
from utils.functions import phase_retrieval
from utils.psf import calculate_psf


def sample_pupil(itr, data, chain, psf_stack, chol_a, chol_phi, defocus_k, mask, del_x, accept_j, x_offset_phase, y_offset_phase, tmp_phase, a_disk, z0, zx, zy, zz, sub_pixel_zeros, start_ind, end_ind, sub_pixel, tp):
    """
    Sample the pupil function parameters.

    Args:
        itr (int): Current iteration number.
        data (ndarray): Observed data.
        chain (dict): Dictionary containing the current state of the Markov chain.
        psf_stack (ndarray): Stack of PSF images.
        chol_a (ndarray): Cholesky decomposition of the covariance matrix for magnitude.
        chol_phi (ndarray): Cholesky decomposition of the covariance matrix for phase.
        defocus_k (float): Defocus parameter.
        mask (ndarray): Binary mask.
        del_x (ndarray): Array of offsets for x, y, and z coordinates.
        accept_j (int): Counter for accepted proposals.
        x_offset_phase (float): Offset for x coordinate in phase retrieval.
        y_offset_phase (float): Offset for y coordinate in phase retrieval.
        tmp_phase (ndarray): Temporary array for phase retrieval.
        a_disk (ndarray): Disk-shaped array for phase retrieval.
        z0 (float): Initial defocus value.
        zx (float): Defocus value in x direction.
        zy (float): Defocus value in y direction.
        zz (float): Defocus value in z direction.
        sub_pixel_zeros (ndarray): Array of sub-pixel zeros.
        start_ind (int): Start index for sub-pixel sampling.
        end_ind (int): End index for sub-pixel sampling.
        sub_pixel (float): Sub-pixel value.
        tp (float): Temperature parameter.

    Returns:
        tuple: Tuple containing the updated magnitude, phase, temporary phase, PSF stack, acceptance counter, and log prior probabilities for magnitude and phase.
    """

    # Extract current state from the Markov chain
    current_phase = chain['Phase']
    current_mag = chain['Mag']
    current_bg = chain['Bg']
    current_i_val = chain['I']
    sig = 0.005
    current_x_val = chain['X']
    current_y_val = chain['Y']
    current_z_val = chain['Z']
    loads = chain['loads']

    # Propose a new phase value
    proposed_phase = current_phase + sig * np.matmul(np.random.normal(size=current_phase.shape), chol_phi)
    sub_phase = phase_retrieval(itr, proposed_phase.copy(), z0, zx, zy, zz, mask, a_disk)

    # Sample new PSF using new Phase
    proposed_psf = calculate_psf(current_mag, sub_phase, current_bg, current_i_val, loads, defocus_k, current_z_val + del_x[:, 2], mask, sub_pixel_zeros, start_ind, end_ind, sub_pixel, current_x_val + del_x[:, 0], current_y_val + del_x[:, 1], x_offset_phase, y_offset_phase)

    # Accept or reject new Phase
    d_log_l = np.sum(data * (np.log(proposed_psf) - np.log(psf_stack)) - (proposed_psf - psf_stack)) / tp
    if d_log_l > np.log(np.random.rand()):
        current_phase[:] = proposed_phase
        tmp_phase[:] = sub_phase
        psf_stack[:] = proposed_psf
        accept_j += 1

    # Sample new Mag
    proposed_mag = np.exp(np.log(current_mag.flatten()) + 0.2 * sig * np.matmul(np.random.normal(size=current_phase.shape), chol_a))
    proposed_mag = proposed_mag.reshape(*mask.shape)

    # Sample new PSF using new Mag
    proposed_psf = calculate_psf(proposed_mag, tmp_phase, current_bg, current_i_val, loads, defocus_k, current_z_val + del_x[:, 2], mask, sub_pixel_zeros, start_ind, end_ind, sub_pixel, current_x_val + del_x[:, 0], current_y_val + del_x[:, 1], x_offset_phase, y_offset_phase)

    # Accept or reject new Mag
    d_log_l = np.sum(data * (np.log(proposed_psf) - np.log(psf_stack)) - (proposed_psf - psf_stack)) / tp
    log_prior_prop = 0
    log_prior_old = 0
    log_prior_ratio = log_prior_prop - log_prior_old
    log_post_ratio = d_log_l + log_prior_ratio
    if log_post_ratio > np.log(np.random.rand()):
        current_mag[:] = proposed_mag
        psf_stack[:] = proposed_psf
        accept_j += 1

    # Compute log prior probabilities for A and Phi
    l_prior_a = 0
    l_prior_phi = 0

    return current_mag, current_phase, tmp_phase, psf_stack, accept_j, l_prior_a, l_prior_phi
