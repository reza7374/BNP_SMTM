import numpy as np
from scipy.spatial.distance import cdist

def cal_cov(xg, yg, t, l, kernel='Exponential'):
    """
    Calculates the covariance matrix between two Gaussian distributions.

    Parameters
    ----------
    xg : np.ndarray
        The x-coordinates of the Gaussian distribution.
    yg : np.ndarray
        The y-coordinates of the Gaussian distribution.
    t : float
        The scale parameter of the Gaussian distribution.
    l : float
        The length scale parameter of the Gaussian distribution.
    kernel : str, optional
        The kernel function to use for the covariance matrix.
        Options are 'Exponential' or 'Quadratic'.
        The default is 'Exponential'.

    Returns
    -------
    np.ndarray
        The covariance matrix between the two Gaussian distributions.

    """
    xg = np.array(xg).flatten()
    yg = np.array(yg).flatten()
    x1 = np.array([xg, yg]).transpose()
    dist = cdist(x1, x1)

    if kernel == 'Exponential':
        k = (t**2) * np.exp(-dist**2 / (2 * l**2))
    elif kernel == 'Quadratic':
        k = (t**2) * (1 + dist**2 / (2 * l**2))

    return k
