"""Utility functions for ABCD RF fitting operations.

This module provides helper functions for complex function fitting, signal processing,
and data manipulation used throughout the ABCD RF fitting process.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares


def complex_fit(
    f: callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper around scipy least_squares for complex function fitting.

    This function enables fitting of complex-valued functions by splitting
    the complex residuals into real and imaginary components for the
    least squares optimization.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). Must take the independent variable
        as the first argument and parameters to fit as separate remaining arguments.
    xdata : np.ndarray
        The independent variable where the data is measured.
    ydata : np.ndarray
        The dependent complex data to fit.
    p0 : np.ndarray, optional
        Initial guess for parameters.
    weights : np.ndarray, optional
        Optional weighting in the calculation of the cost function.
    **kwargs : Any
        Additional arguments passed to scipy.optimize.least_squares.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Optimal parameters
        - Parameter covariance matrix

    Raises
    ------
    ValueError
        If ydata length is not greater than the number of parameters.
    """
    if (np.array(ydata).size - len(p0)) <= 0:
        raise ValueError(
            "yData length should be greater than the number of parameters."
        )

    def residuals(params, x, y):
        """Compute the residual for the least square algorithm."""
        diff = weights * f(x, *params) - y if weights is not None else f(x, *params) - y
        flat_diff = np.zeros(diff.size * 2, dtype=np.float64)
        flat_diff[0 : flat_diff.size : 2] = diff.real
        flat_diff[1 : flat_diff.size : 2] = diff.imag
        return flat_diff

    kwargs_ls = kwargs.copy()
    kwargs_ls.setdefault("max_nfev", 1000)
    kwargs_ls.setdefault("ftol", 1e-2)
    opt_res = least_squares(residuals, p0, args=(xdata, ydata), **kwargs_ls)

    jac = opt_res.jac
    cost = opt_res.cost

    pcov = np.linalg.inv(jac.T.dot(jac))
    pcov *= cost / (np.array(ydata).size - len(p0))

    popt = opt_res.x

    return popt, pcov


def guess_edelay_from_gradient(
    freq: np.ndarray, signal: np.ndarray, n: int = -1
) -> float:
    """Estimate electrical delay from phase gradient across frequency.

    This function estimates the electrical delay by computing the mean
    phase difference between the beginning and end of the frequency sweep.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array in Hz.
    signal : np.ndarray
        Complex signal array.
    n : int, optional
        Number of points to use from each end (default: -1, uses all points).

    Returns
    -------
    float
        Estimated electrical delay in seconds.
    """
    dtheta = np.mean(np.angle(signal[-n:] / zeros2eps(signal[:n])))
    df = np.mean(np.diff(freq))

    return dtheta / df / 2 / np.pi


def smooth_gradient(signal: np.ndarray) -> np.ndarray:
    """Compute smoothed gradient of a signal using Gaussian derivative kernel.

    This function applies a Gaussian derivative convolution to compute
    a smoothed version of the signal gradient, which is used for
    weighting in the ABCD fitting algorithm.

    Parameters
    ----------
    signal : np.ndarray
        Input complex signal array.

    Returns
    -------
    np.ndarray
        Smoothed gradient of the input signal.
    """

    def dnormaldx(x, x_0, sigma):
        return -(x - x_0) * np.exp(-0.5 * ((x - x_0) / sigma) ** 2)

    conv_kernel_size = max(min(100, signal.size // 20), 2)

    conv_kernel = dnormaldx(
        x=np.arange(0.5, conv_kernel_size + 0.5, 1),
        x_0=conv_kernel_size / 2,
        sigma=conv_kernel_size / 8,
    )

    gradient = np.convolve(signal, conv_kernel, "same")
    gradient[: conv_kernel_size // 2] = gradient[
        conv_kernel_size // 2 : 2 * (conv_kernel_size // 2)
    ][::-1]
    gradient[-(conv_kernel_size // 2) :] = gradient[
        -2 * (conv_kernel_size // 2) : -(conv_kernel_size // 2)
    ][::-1]

    return gradient


eps = np.finfo(float).eps


def zeros2eps(x: Union[float, complex, np.ndarray]) -> np.ndarray:
    """Replace zeros with machine epsilon to avoid division by zero.

    This utility function replaces any zero values in the input with
    the machine epsilon to prevent numerical issues in calculations.

    Parameters
    ----------
    x : float, complex, or np.ndarray
        Input value or array.

    Returns
    -------
    np.ndarray
        Array with zeros replaced by machine epsilon.
    """
    y = np.array(x)
    y[np.abs(y) < eps] = eps

    return y


def dB(x: Union[float, complex, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert magnitude to decibels.

    Parameters
    ----------
    x : float, complex, or np.ndarray
        Input value(s) to convert.

    Returns
    -------
    float or np.ndarray
        Magnitude in decibels (20*log10(|x|)).
    """
    return 20 * np.log10(np.abs(x))


def deg(x: Union[float, complex, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert phase from radians to degrees.

    Parameters
    ----------
    x : float, complex, or np.ndarray
        Input complex value(s).

    Returns
    -------
    float or np.ndarray
        Phase in degrees.
    """
    return np.angle(x) * 180 / np.pi


def get_prefix(x: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], str]:
    """Get appropriate SI prefix for a value.

    Determines the appropriate SI unit prefix (m, k, M, G, etc.) for a given
    value and returns the scaled value along with the prefix string.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) to get prefix for.

    Returns
    -------
    tuple[Union[float, np.ndarray], str]
        A tuple containing:
        - Scaled value
        - SI prefix string
    """
    prefix = [
        "y",  # yocto
        "z",  # zepto
        "a",  # atto
        "f",  # femto
        "p",  # pico
        "n",  # nano
        "u",  # micro
        "m",  # mili
        "",
        "k",  # kilo
        "M",  # mega
        "G",  # giga
        "T",  # tera
        "P",  # peta
        "E",  # exa
        "Z",  # zetta
        "Y",  # yotta
    ]

    max_x = np.abs(np.max(x))

    if max_x > 10 * eps:
        index = int(np.log10(max_x) / 3 + 8)
        return (x * 10 ** (-3 * (index - 8)), prefix[index])

    return (0, "")


def get_prefix_str(x: Union[float, np.ndarray], precision: int = 2) -> str:
    """Format a value with appropriate SI prefix as a string.

    Combines the functionality of get_prefix() with string formatting
    to produce a human-readable representation of a value with SI units.

    Parameters
    ----------
    x : float or np.ndarray
        Input value to format.
    precision : int, optional
        Number of decimal places (default: 2).

    Returns
    -------
    str
        Formatted string with value and SI prefix.

    Examples
    --------
    >>> get_prefix_str(1500)
    '1.50 k'
    >>> get_prefix_str(0.001)
    '1.00 m'
    """
    return f"%.{precision}f %s" % get_prefix(x)
