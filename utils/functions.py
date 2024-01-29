from math import gamma

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
    return (b**a / gamma(a)) * x**(-a-1) * np.exp(-b / x)
