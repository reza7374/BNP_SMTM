import numpy as np
from math import gamma
from scipy.special import gammaln
import scipy.stats as st

def gampdf(x: float, k: float, theta: float) -> float:
    """
    Calculates the gamma probability density function at point x.

    Args:
        x (float): point at which to evaluate the PDF
        k (float): shape parameter of the gamma distribution
        theta (float): scale parameter of the gamma distribution

    Returns:
        float: value of the gamma PDF at point x

    Raises:
        ValueError: if x is not a float
        ValueError: if k is not a float
        ValueError: if theta is not a float
        ValueError: if k is not greater than zero
        ValueError: if theta is not greater than zero
    """
    if not isinstance(x, float):
        raise ValueError("x must be a float")
    if not isinstance(k, float):
        raise ValueError("k must be a float")
    if not isinstance(theta, float):
        raise ValueError("theta must be a float")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if theta <= 0:
        raise ValueError("theta must be greater than zero")

    return st.gamma.pdf(x, a=k, scale=theta)


def inv_gamma_pdf(x, a, b):
    """
    Calculate the inverse gamma probability density function (PDF) at a given point x.

    Args:
        x (float): Point at which to evaluate the PDF.
        a (float): Shape parameter of the inverse gamma distribution.
        b (float): Scale parameter of the inverse gamma distribution.

    Returns:
        float: Value of the inverse gamma PDF at the specified point x.
    """
    return np.exp(a * np.log(b) - gammaln(a) - (a + 1) * np.log(x) - b / x)

def norm_pdf(x, mu, sigma):
    """
    Calculates the normal probability density function at point x.

    Args:
        x (float): point at which to evaluate the PDF
        mu (float): mean of the normal distribution
        sigma (float): standard deviation of the normal distribution

    Returns:
        float: value of the normal PDF at point x
    """
    return st.norm.pdf(x, mu, sigma)

def bernoulli_pdf(x_values, success_prob, failure_prob):
    """
    Calculates the Bernoulli probability for a series of values x_values, given the success probability and failure probability.

    Args:
        x_values (numpy.ndarray): array of values to calculate the probability for
        success_prob (float): probability of success
        failure_prob (float): probability of failure

    Returns:
        numpy.ndarray: array of Bernoulli probabilities for each value in x_values

    Raises:
        ValueError: if the input arrays are not 1-dimensional or of the same length
    """
    if not isinstance(x_values, np.ndarray):
        raise ValueError("x_values must be a numpy array")
    if x_values.ndim != 1 or x_values.size != len(success_prob):
        raise ValueError("x_values must be a 1-dimensional array of the same length as success_prob")

    param = 1 / (1 + ((success_prob - 1) / failure_prob))
    probability = np.where(x_values == 0, 1 - param, param)

    return probability

def phase_retrieval(iteration, phase_estimate, z0, zx, zy, zz, mask, adisk):
    # create a copy of the original phase estimate
    updated_phase = np.copy(phase_estimate)
    
    # iterate the phase retrieval algorithm
    for _ in range(iteration):
        # compute the shift in x, y, z, and offset
        x_shift = np.sum(updated_phase * zx * mask) / adisk
        y_shift = np.sum(updated_phase * zy * mask) / adisk
        z_shift = np.sum(updated_phase * zz * mask) / adisk
        offset = np.sum(updated_phase * z0 * mask) / adisk
        
        # subtract the shifts and offset from the phase estimate, and apply the mask
        updated_phase = (updated_phase - x_shift * zx - y_shift * zy - z_shift * zz - offset * z0) * mask

    return updated_phase


