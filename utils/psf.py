import numpy as np


def calculate_psf(pupil_magnitude, pupil_phase, background_intensity, object_intensity, loads, defocus_coefficient, z_values, mask, subpixel_zeros, start_index, end_index, subpixel_resolution, x_values, y_values, x_offset_phase='Not Given', y_offset_phase='Not Given'):
    """
    Calculate the Point Spread Function (PSF) for a given set of parameters.

    Args:
        pupil_magnitude (ndarray): Magnitude of the pupil function.
        pupil_phase (ndarray): Phase of the pupil function.
        background_intensity (float): Background intensity.
        object_intensity (float): Object intensity.
        loads (ndarray): Array indicating which loads to calculate PSF for.
        defocus_coefficient (float): Defocus coefficient.
        z_values (ndarray): Array of defocus values.
        mask (ndarray): Mask for the pupil function.
        subpixel_zeros (ndarray): Array for padding zeros for subpixel resolution.
        start_index (int): Start index for subpixel resolution.
        end_index (int): End index for subpixel resolution.
        subpixel_resolution (float): Subpixel resolution.
        x_values (ndarray): Array of x values.
        y_values (ndarray): Array of y values.
        x_offset_phase (str, optional): X offset phase. Defaults to 'Not Given'.
        y_offset_phase (str, optional): Y offset phase. Defaults to 'Not Given'.

    Returns:
        tuple: Tuple containing the background PSF and the PSF.
    """
    num_pixels = mask.shape[1]
    psf = np.zeros((num_pixels, num_pixels, z_values.shape[0], z_values.shape[1]))
    bg_psf = np.zeros((num_pixels, num_pixels, z_values.shape[0]))

    pupil_phase_x_offset_phase = pupil_phase + x_offset_phase * x_values + y_offset_phase * y_values
    otf = mask * pupil_magnitude * np.exp(1j * pupil_phase_x_offset_phase)

    loads_indices = np.where(loads == 1)[0]
    for nn in loads_indices:
        for zz in range(z_values.shape[0]):
            defocus_phase = defocus_coefficient * z_values[zz, nn]
            phase = mask * (defocus_phase + pupil_phase_x_offset_phase[zz, nn])
            # Optical transfer function (propagator)
            otf = mask * pupil_magnitude * np.exp(1j * phase)
            # Parseval Normalization
            norm = np.sqrt(np.sum(np.abs(otf))) * num_pixels
            # Padding zeros for subpixel resolution
            subpixel_zeros[start_index:end_index, start_index:end_index] = otf
            tmp_psf = (np.abs(np.fft.fftshift(np.fft.fft2(subpixel_zeros / norm)))) ** 2
            psf[:, :, zz, nn] = tmp_psf
            psf[:, :, zz, nn] /= np.sum(psf[:, :, zz, nn])

    psf = np.sum(psf, axis=3)
    bg_psf = object_intensity * psf + background_intensity
    return bg_psf, psf
