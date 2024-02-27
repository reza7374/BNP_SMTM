# filepath: /home/reza/software/bnp_smtm/utils/covariance_calc.py
import numpy as np
from scipy.spatial.distance import cdist


def calculate_covariance(x_values, y_values, t, l, kernel='exponential'):
    """
    Calculate the covariance matrix based on the given x and y values.

    Args:
        x_values (array-like): The x values.
        y_values (array-like): The y values.
        t (float): The scaling factor.
        l (float): The length scale.
        kernel (str, optional): The kernel function to use. Defaults to 'exponential'.

    Returns:
        ndarray: The covariance matrix.
    """
    # Convert x and y values to coordinates
    coordinates = np.column_stack((np.array(x_values).flatten(), np.array(y_values).flatten()))
    
    # Calculate pairwise distances between coordinates
    distances = cdist(coordinates, coordinates)
    
    if kernel == 'exponential':
        # Calculate covariance using exponential kernel
        covariance = (t ** 2) * np.exp(-distances ** 2 / l ** 2 / 2)
    elif kernel == 'quadratic':
        # Calculate covariance using quadratic kernel
        covariance = (t ** 2) * (1 + distances ** 2 / l ** 2 / 2)
    
    return covariance
