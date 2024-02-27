import numpy as np
from utils.functions import norm_pdf


def switch_label_interval(chain, struct, bnp, accept_count):
    """
    Switches the labels of particles in an interval of frames in the chain based on certain conditions.

    Args:
        chain (dict): Dictionary containing the chain information.
        struct (dict): Dictionary containing the structural information.
        bnp (dict): Dictionary containing the BNP information.
        accept_count (int): Number of accepted switches.

    Returns:
        tuple: Tuple containing the updated positions of particles in the X, Y, and Z dimensions,
               the updated accept count, and the log prior probability.
    """
    positions_x = chain['X']
    positions_y = chain['Y']
    positions_z = chain['Z']
    loads = chain['loads']
    log_p = 0
    
    # Check if there are at least two active particles
    if np.sum(loads) > 1:
        diffusion_coefficient = chain['D']
        time_step = bnp['Dt']
        sig_d = np.sqrt(2 * diffusion_coefficient * time_step)
        probabilities = loads / np.sum(loads)
        sig_x = struct['NPix'] * struct['PixelSize'] / 2
        
        size_z = positions_z.shape
        frame = np.random.choice(size_z[0], size=2, replace=False)
        
        num_particles = np.random.choice(len(loads), size=2, p=probabilities, replace=False)
        
        # Swap the positions of particles in the selected frame and with the selected particles
        positions_x[frame[0]:frame[1], num_particles[0]], positions_x[frame[0]:frame[1], num_particles[1]] = positions_x[frame[0]:frame[1], num_particles[1]], positions_x[frame[0]:frame[1], num_particles[0]]
        positions_y[frame[0]:frame[1], num_particles[0]], positions_y[frame[0]:frame[1], num_particles[1]] = positions_y[frame[0]:frame[1], num_particles[1]], positions_y[frame[0]:frame[1], num_particles[0]]
        positions_z[frame[0]:frame[1], num_particles[0]], positions_z[frame[0]:frame[1], num_particles[1]] = positions_z[frame[0]:frame[1], num_particles[1]], positions_z[frame[0]:frame[1], num_particles[0]]
        
        # Calculate log prior fusion for each dimension
        log_prior_diffusion_1_x = np.sum(np.log(norm_pdf(positions_x[0, num_particles], 0, sig_x)) - np.log(norm_pdf(positions_x[0, num_particles], 0, sig_x)))
        log_prior_diffusion_1_y = np.sum(np.log(norm_pdf(positions_y[0, num_particles], 0, sig_x)) - np.log(norm_pdf(positions_y[0, num_particles], 0, sig_x)))
        log_prior_diffusion_1_z = np.sum(np.log(norm_pdf(positions_z[0, num_particles], 0, sig_x)) - np.log(norm_pdf(positions_z[0, num_particles], 0, sig_x)))
        log_prior_diffusion_x = np.sum(np.log(norm_pdf(positions_x[1:, num_particles], positions_x[:-1, num_particles], sig_d)) - np.log(norm_pdf(positions_x[1:, num_particles], positions_x[:-1, num_particles], sig_d)))
        log_prior_diffusion_y = np.sum(np.log(norm_pdf(positions_y[1:, num_particles], positions_y[:-1, num_particles], sig_d)) - np.log(norm_pdf(positions_y[1:, num_particles], positions_y[:-1, num_particles], sig_d)))
        log_prior_diffusion_z = np.sum(np.log(norm_pdf(positions_z[1:, num_particles], positions_z[:-1, num_particles], sig_d)) - np.log(norm_pdf(positions_z[1:, num_particles], positions_z[:-1, num_particles], sig_d)))
        log_posterior = np.logaddexp.reduce([log_prior_diffusion_1_x, log_prior_diffusion_1_y, log_prior_diffusion_1_z, log_prior_diffusion_x, log_prior_diffusion_y, log_prior_diffusion_z])
        
        if log_posterior > np.log(np.random.rand()):
            accept_count += 1
            log_p = log_posterior
    
    return positions_x, positions_y, positions_z, accept_count, log_p


def switch_label_one_frame(chain, struct, bnp, accept_count):
    """
    Switches the labels of particles in one frame of the chain based on certain conditions.

    Args:
        chain (dict): Dictionary containing the chain information.
        struct (dict): Dictionary containing the structural information.
        bnp (dict): Dictionary containing the BNP information.
        accept_count (int): Number of accepted switches.

    Returns:
        tuple: Tuple containing the updated positions of particles in the X, Y, and Z dimensions,
               the updated accept count, and the log prior probability.
    """
    positions_x = chain['X']
    positions_y = chain['Y']
    positions_z = chain['Z']
    loads = chain['loads']
    log_p = 0
    
    # Check if there are at least two active particles
    if np.sum(loads) > 1:
        diffusion_coefficient = chain['D']
        time_step = bnp['Dt']
        sig_d = np.sqrt(2 * diffusion_coefficient * time_step)
        probabilities = loads / np.sum(loads)
        sig_x = struct['NPix'] * struct['PixelSize'] / 2
        
        size_z = positions_z.shape
        frame = np.random.randint(size_z[0])
        
        while True:
            num_particles = np.random.choice(len(loads), size=2, p=probabilities)
            if num_particles[0] != num_particles[1]:
                break
        
        temp_x = positions_x.copy()
        temp_y = positions_y.copy()
        temp_z = positions_z.copy()
        
        temp_x[frame, num_particles[0]], temp_x[frame, num_particles[1]] = temp_x[frame, num_particles[1]], temp_x[frame, num_particles[0]]
        temp_y[frame, num_particles[0]], temp_y[frame, num_particles[1]] = temp_y[frame, num_particles[1]], temp_y[frame, num_particles[0]]
        temp_z[frame, num_particles[0]], temp_z[frame, num_particles[1]] = temp_z[frame, num_particles[1]], temp_z[frame, num_particles[0]]
        
        log_prior_diffusion_1_x = np.sum(np.log(norm_pdf(temp_x[0, num_particles], 0, sig_x)) - np.log(norm_pdf(positions_x[0, num_particles], 0, sig_x)))
        log_prior_diffusion_1_y = np.sum(np.log(norm_pdf(temp_y[0, num_particles], 0, sig_x)) - np.log(norm_pdf(positions_y[0, num_particles], 0, sig_x)))
        log_prior_diffusion_1_z = np.sum(np.log(norm_pdf(temp_z[0, num_particles], 0, sig_x)) - np.log(norm_pdf(positions_z[0, num_particles], 0, sig_x)))
        log_prior_diffusion_x = np.sum(np.log(norm_pdf(temp_x[1:, num_particles], temp_x[:-1, num_particles], sig_d)) - np.log(norm_pdf(positions_x[1:, num_particles], positions_x[:-1, num_particles], sig_d)))
        log_prior_diffusion_y = np.sum(np.log(norm_pdf(temp_y[1:, num_particles], temp_y[:-1, num_particles], sig_d)) - np.log(norm_pdf(positions_y[1:, num_particles], positions_y[:-1, num_particles], sig_d)))
        log_prior_diffusion_z = np.sum(np.log(norm_pdf(temp_z[1:, num_particles], temp_z[:-1, num_particles], sig_d)) - np.log(norm_pdf(positions_z[1:, num_particles], positions_z[:-1, num_particles], sig_d)))
        
        log_posterior = log_prior_diffusion_1_x + log_prior_diffusion_1_y + log_prior_diffusion_1_z + log_prior_diffusion_x + log_prior_diffusion_y + log_prior_diffusion_z
        
        if log_posterior > np.log(np.random.rand()):
            positions_x[frame, num_particles[0]], positions_x[frame, num_particles[1]] = positions_x[frame, num_particles[1]], positions_x[frame, num_particles[0]]
            positions_y[frame, num_particles[0]], positions_y[frame, num_particles[1]] = positions_y[frame, num_particles[1]], positions_y[frame, num_particles[0]]
            positions_z[frame, num_particles[0]], positions_z[frame, num_particles[1]] = positions_z[frame, num_particles[1]], positions_z[frame, num_particles[0]]
            accept_count += 1
            log_p = log_posterior
    
    return positions_x, positions_y, positions_z, accept_count, log_p
