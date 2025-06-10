import numpy as np
from scipy.optimize import least_squares


def complex_fit(f, xdata, ydata, p0=None, weights=None, **kwargs):
    """
    Wrapper around scipy least_square for complex functions

    Args:
        f: The model function, f(x, …). It must take the independent variable
            as the first argument and the parameters to fit as separate
            remaining arguments.
        xdata: The independent variable where the data is measured.
        ydata: The dependent data.
        p0: Initial guess on independent variables.
        weights: Optional weighting in the calculation of the cost function.
        kwargs: passed to the leas_square function

    Returns
    -------
        A tuple with the optimal parameters and the covariance matrix
    """
    if (np.array(ydata).size - len(p0)) <= 0:
        raise ValueError(
            "yData length should be greater than the number of parameters."
        )

    def residuals(params, x, y):
        """Computes the residual for the least square algorithm"""
        if weights is not None:
            diff = weights * f(x, *params) - y
        else:
            diff = f(x, *params) - y
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


def guess_edelay_from_gradient(freq, signal, n=-1):

    dtheta = np.mean(np.angle(signal[-n:] / zeros2eps(signal[:n])))
    df = np.mean(np.diff(freq))

    return dtheta / df / 2 / np.pi


def smooth_gradient(signal):
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


def zeros2eps(x):
    """
    args:
        x: float, complex, or numpy array

    return:
        y: numpy array

    replace the zeros of a float or numpy array bien the smallest float number
    """
    y = np.array(x)
    y[np.abs(y) < eps] = eps

    return y


def dB(x):

    return 20 * np.log10(np.abs(x))


def deg(x):

    return np.angle(x) * 180 / np.pi


def get_prefix(x):

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


def get_prefix_str(x, precision=2):

    return f"%.{precision}f %s" % get_prefix(x)
