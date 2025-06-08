"""ABCD RF resonator parameter extraction module.

This module provides core functionality for fitting resonator scattering parameters
using an optimized rational function approach with iterative weight refinement.

The main workflow is:
1. Extract ABCD coefficients using `get_abcd()`
2. Convert to physical parameters using `abcd2params()`
3. Perform final optimization using `fit_signal()`

Typical usage:
    >>> import numpy as np
    >>> from abcd_rf_fit import fit_signal
    >>>
    >>> # Load your S-parameter data
    >>> freq = np.linspace(4e9, 6e9, 1001)  # Hz
    >>> s21_data = load_s_parameters()  # Complex array
    >>>
    >>> # Fit the data
    >>> fit_func, result = fit_signal(freq, s21_data, "transmission")
    >>> print(f"Resonance frequency: {result.f_0:.6e} Hz")
    >>> print(f"Quality factor: {result.f_0/result.kappa:.1f}")
"""

import warnings
from typing import List, Tuple

import numpy as np

from .resonators import (
    FitResult,
    ResonatorParams,
    hanger,
    hanger_mismatched,
    reflection,
    reflection_mismatched,
    resonator_dict,
    transmission,
)
from .utils import (
    complex_fit,
    guess_edelay_from_gradient,
    smooth_gradient,
)


def get_abcd(
    freq: np.ndarray, signal: np.ndarray, rec_depth: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ABCD coefficients for rational function fit: S(ω) = (a+b*ω)/(c+d*ω).

    This function implements an iterative algorithm that refines the weighting
    in the least squares regression to improve the quality of the rational function fit.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array in Hz.
    signal : np.ndarray
        Complex signal array (S-parameters).
    rec_depth : int, optional
        Recursion depth for iterative weight refinement (default: 0).
        - 0: Single pass, fastest, good for clean data
        - 1: One refinement iteration (recommended)
        - 2+: Multiple iterations, slower, for noisy data

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - abcd_coefficients: Array [a, b, c, d] for rational function
        - fitted_signal: Fitted signal values at input frequencies

    Notes
    -----
    -----    The algorithm performs iterative refinement of the gradient estimate
    used as weights in the least squares regression. Higher rec_depth
    doesn't always mean better fits - convergence typically occurs within
    1-2 iterations.
    """
    freq_center = np.mean(freq)

    x_design = np.ones((2, freq.size))
    x_design[1, :] = freq - freq_center

    signal_grad = smooth_gradient(signal)

    for _ in range(rec_depth + 1):
        xx = (x_design * np.abs(signal_grad)) @ x_design.T
        xdyx = (signal * x_design * np.abs(signal_grad)) @ x_design.T
        xdycx = (np.conj(signal) * x_design * np.abs(signal_grad)) @ x_design.T
        xdy2x = (np.abs(signal) ** 2 * x_design * np.abs(signal_grad)) @ x_design.T

        up_right = np.linalg.inv(xx) @ xdyx
        bottom_left = np.linalg.inv(xdy2x) @ xdycx

        to_diag = np.zeros((4, 4), dtype=complex)
        to_diag[:2, 2:] = up_right
        to_diag[2:, :2] = bottom_left

        v, w = np.linalg.eig(to_diag)

        abcd = w[:, np.argmin(np.abs(1 - v))]

        signal_grad = (abcd[2] + abcd[3] * (freq - freq_center)) ** -2

    abcd[0::2] -= abcd[1::2] * freq_center

    fit = (abcd[0] + abcd[1] * freq) / (abcd[2] + abcd[3] * freq)

    return abcd, fit


def abcd2params(abcd: np.ndarray, geometry: str) -> List[float]:
    """Convert ABCD coefficients to physical resonator parameters.

    Transforms the rational function coefficients [a, b, c, d] into physically
    meaningful resonator parameters depending on the measurement geometry.

    Parameters
    ----------
    ----------    abcd : np.ndarray
        Array of coefficients [a, b, c, d] from rational function
        S(ω) = (a+b*ω)/(c+d*ω).
    geometry : str
        Measurement geometry string:
        - "r", "reflection": Reflection measurement (S₁₁)
        - "rm", "reflection_mismatched": Reflection with impedance mismatch
        - "t", "transmission": Transmission measurement (S₂₁)
        - "h", "hanger": Hanger coupling measurement
        - "hm", "hanger_mismatched": Hanger with impedance mismatch

    Returns
    -------
    list[float]
        Physical parameters depending on geometry:
        - Reflection: [f_0, kappa, kappa_c_real, Re(a_in), Im(a_in)]
        - Reflection mismatched: [f_0, kappa, kappa_c_real, phi_0, Re(a_in), Im(a_in)]
        - Transmission: [f_0, kappa, Re(a_in), Im(a_in)]
        - Hanger: [f_0, kappa_i, kappa_c, Re(a_in), Im(a_in)]
        - Hanger mismatched: [f_0, kappa_i, kappa_c, phi_0, Re(a_in), Im(a_in)]

    Notes
    -----
    Parameter definitions:
        f_0: Resonance frequency [Hz]
        kappa: Total linewidth [Hz]
        kappa_i: Internal loss rate [Hz]
        kappa_c: Coupling rate [Hz]
        phi_0: Impedance mismatch phase [rad]
        a_in: Input amplitude (complex, stored as real and imaginary parts)
    """
    a, b, c, d = abcd

    if resonator_dict[geometry] == reflection:
        f_0 = -0.5 * np.real(a / b + c / d)
        a_in = b / d

        kappa_c_real = np.real(1j * (c / d - a / b))
        kappa_i = -np.imag(a / b + c / d)
        kappa = kappa_i + kappa_c_real

        return f_0, kappa, kappa_c_real, np.real(a_in), np.imag(a_in)

    if resonator_dict[geometry] == reflection_mismatched:
        f_0 = -np.real(c / d)
        a_in = b / d

        kappa_c_imag = np.real(a / b - c / d)
        kappa_c_real = np.real(1j * (c / d - a / b))
        kappa_i = -np.imag(a / b + c / d)

        kappa = kappa_i + kappa_c_real

        phi_0 = np.angle(kappa_c_real - 1j * kappa_c_imag)

        return f_0, kappa, kappa_c_real, phi_0, np.real(a_in), np.imag(a_in)

    if resonator_dict[geometry] == transmission:
        signal_f_0_before = (a - b * np.real(c / d)) / (c - d * np.real(c / d))
        a, b = a - c * b / d, 0
        signal_f_0_after = (a - b * np.real(c / d)) / (c - d * np.real(c / d))
        a = a * signal_f_0_before / signal_f_0_after

        kappa = -2 * np.imag(c / d)
        f_0 = -np.real(c / d)
        a_in = 2j * a / d

        return f_0, kappa, np.real(a_in), np.imag(a_in)

    if resonator_dict[geometry] == hanger:
        kappa_c_r = np.real(1j * (c / d - a / b))
        kappa_i_r = -np.imag(a / b + c / d)
        f_0 = -0.5 * np.real(a / b + c / d)
        a_in = b / d

        kappa_c_real = 2 * kappa_c_r
        kappa_i = kappa_i_r - kappa_c_r

        kappa = kappa_i + kappa_c_real

        return f_0, kappa, kappa_c_real, np.real(a_in), np.imag(a_in)

    if resonator_dict[geometry] == hanger_mismatched:
        f_0 = -np.real(c / d)
        a_in = b / d

        kappa_c_imag = 2 * np.real(a / b - c / d)
        kappa_c_real = 2 * np.real(1j * (c / d - a / b))
        kappa_i = -2 * np.imag(c / d) - kappa_c_real

        kappa = kappa_i + kappa_c_real

        phi_0 = np.angle(kappa_c_real - 1j * kappa_c_imag)

        return f_0, kappa, kappa_c_real, phi_0, np.real(a_in), np.imag(a_in)
    return None


def get_fit_function(geometry: str, amplitude: bool = True, edelay: bool = True):
    """Create a fit function for the specified resonator geometry and parameters.

    Constructs a function that can be used with optimization routines to fit
    resonator data. The function combines the physical resonator model with
    optional amplitude scaling and electrical delay corrections.

    Parameters
    ----------
    geometry : str
        Resonator measurement geometry:
        - "r", "reflection": Reflection measurement
        - "rm", "reflection_mismatched": Reflection with mismatch
        - "t", "transmission": Transmission measurement
        - "h", "hanger": Hanger coupling
        - "hm", "hanger_mismatched": Hanger with mismatch
    amplitude : bool, optional
        If True, include complex amplitude scaling parameters (default: True).
    edelay : bool, optional
        If True, include electrical delay parameter (default: True).

    Returns
    -------
    callable
        Function that takes (frequency, *parameters) and returns
        the model prediction for the S-parameter data.

    Notes
    -----
    The returned function signature depends on the geometry and options:
    - Base parameters vary by geometry (see abcd2params docstring)
    - If amplitude=True: adds Re(a_in), Im(a_in) parameters
    - If edelay=True: adds electrical delay parameter [s]
    """
    resonator_func = resonator_dict[geometry]

    if not amplitude and not edelay:
        return resonator_func

    if amplitude and not edelay:

        def fit_func(*args):
            return resonator_func(*args[:-2]) * (args[-2] + 1j * args[-1])

        return fit_func

    if not amplitude and edelay:

        def fit_func(*args):
            return resonator_func(*args[:-1]) * np.exp(2j * np.pi * args[-1] * args[0])

        return fit_func

    if amplitude and edelay:

        def fit_func(*args):
            return (
                resonator_func(*args[:-3])
                * (args[-3] + 1j * args[-2])
                * np.exp(2j * np.pi * args[-1] * args[0])
            )

        return fit_func

    raise Exception("Unreachable")


def meta_fit_edelay(freq: np.ndarray, signal: np.ndarray, rec_depth: int = 0) -> float:
    """Estimate electrical delay by minimizing fit residuals across delay values.

    This function performs a grid search over electrical delay values around
    an initial gradient-based estimate to find the delay that minimizes the
    L2 error of the rational function fit.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array in Hz.
    signal : np.ndarray
        Complex signal array (S-parameters).
    rec_depth : int, optional
        Recursion depth for ABCD fit (default: 0).

    Returns
    -------
    float
        Optimal electrical delay in seconds.

    Notes
    -----
    The search spans ±1.5/(freq_max - freq_min) around the gradient estimate,
    using 21 equally spaced test points for robust delay estimation.
    """
    quick_fit = get_abcd

    guess_edelay = guess_edelay_from_gradient(freq, signal)

    edelay_span = 1.5 / (np.max(freq) - np.min(freq))

    edelay_array = guess_edelay + np.linspace(-1, 1, 21) * edelay_span
    l2_error_array = np.zeros_like(edelay_array)

    for i, ed in enumerate(edelay_array):
        s = signal * np.exp(-2j * np.pi * freq * ed)
        _, abcd_fit = quick_fit(freq, s, rec_depth)

        l2_error_array[i] = np.sum(np.abs(s - abcd_fit) ** 2) / freq.size

    return edelay_array[np.argmin(l2_error_array)]


def _fit_signal_core(
    freq: np.ndarray,
    signal: np.ndarray,
    geometry: str,
    fit_amplitude: bool = True,
    fit_edelay: bool = True,
    final_ls_opti: bool = True,
    allow_mismatch: bool = True,
    rec_depth: int = 1,
    suppress_warnings: bool = False,
) -> FitResult:
    """Core fitting implementation without API redundancy.
    
    This is the internal implementation that returns only a FitResult object.
    """
    edelay = meta_fit_edelay(freq, signal, rec_depth) if fit_edelay else 0

    if resonator_dict[geometry] == reflection and allow_mismatch:
        geometry = "rm"
    elif resonator_dict[geometry] == hanger and allow_mismatch:
        geometry = "hm"

    corrected_signal = signal * np.exp(-2j * np.pi * edelay * freq)

    quick_fit = get_abcd

    abcd, _ = quick_fit(freq, corrected_signal, rec_depth)

    params = abcd2params(abcd, geometry)
    if not fit_amplitude:
        params = params[:-2]
    if fit_edelay:
        params = [*params, edelay]

    fit_func = get_fit_function(geometry, fit_amplitude, fit_edelay)

    pcov = None
    if final_ls_opti:
        params, pcov = complex_fit(fit_func, freq, signal, params)

    resonator_params = ResonatorParams(params, geometry, freq, signal)

    if resonator_params.phi_0 is not None and np.abs(resonator_params.phi_0) > 0.25:
        if not suppress_warnings:
            warnings.warn(
                "Extracted phi_0 greater than 0.25, this might indicate a big "
                "impedance mismatch, values of kappa_i and kappa_c might be "
                "affected, you can try to set: allow_mismatch=False",
                UserWarning,
                stacklevel=2,
            )    # Return FitResult object with covariance matrix information
    fit_result = FitResult(params, geometry, freq, signal, pcov, fit_func)
    return fit_result


def fit_signal(
    freq: np.ndarray,
    signal: np.ndarray,
    geometry: str,
    fit_amplitude: bool = True,
    fit_edelay: bool = True,
    final_ls_opti: bool = True,
    allow_mismatch: bool = True,
    rec_depth: int = 1,
    api_warning: bool = True,
    suppress_warnings: bool = False,
) -> Tuple[callable, FitResult]:
    """Fit resonator S-parameter data to extract physical parameters.

    **DEPRECATED**: This function returns a redundant tuple (fit_func, FitResult)
    where FitResult.fit_func already contains the same function. 
    Use analyze() instead for a cleaner API.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array in Hz.
    signal : np.ndarray
        Complex S-parameter data to fit.
    geometry : str
        Resonator measurement geometry:
        - "r", "reflection": Reflection measurement (S₁₁)
        - "rm", "reflection_mismatched": Reflection with mismatch
        - "t", "transmission": Transmission measurement (S₂₁)
        - "h", "hanger": Hanger coupling measurement
        - "hm", "hanger_mismatched": Hanger with mismatch
    fit_amplitude : bool, optional
        If True, fit complex amplitude scaling (default: True).
    fit_edelay : bool, optional
        If True, estimate and fit electrical delay (default: True).
    final_ls_opti : bool, optional
        If True, perform final nonlinear optimization (default: True).
    allow_mismatch : bool, optional
        If True, automatically use mismatched models when appropriate (default: True).
    rec_depth : int, optional
        ABCD algorithm recursion depth, 0-2 recommended (default: 1).
    api_warning : bool, optional
        If True, show deprecation warning (default: True).
    suppress_warnings : bool, optional
        If True, suppress impedance mismatch warnings (default: False).

    Returns
    -------
    tuple[callable, FitResult]
        A tuple containing:
        - fit_function: Callable that evaluates the fitted model
        - fit_result: FitResult object with parameters, uncertainties,
          and quality metrics
        
        NOTE: This is redundant since fit_result.fit_func contains the same function.

    Notes
    -----
    **DEPRECATED**: Use analyze() instead which returns only FitResult.
    The FitResult object provides access to:
    - Parameter values and uncertainties (if final_ls_opti=True)
    - Goodness of fit metrics (R², reduced χ², RMS residual)
    - Fit validation with warnings and recommendations
    - Parameter correlation matrix
    - Save/load functionality for analysis results

    Examples
    --------
    >>> # OLD (deprecated):
    >>> fit_func, result = fit_signal(freq, s21_data, "transmission")
    >>> 
    >>> # NEW (recommended):
    >>> result = analyze(freq, s21_data, "transmission")
    >>> fit_func = result.fit_func  # Same function available as attribute
    """
    if api_warning:
        warnings.warn(
            "fit_signal() is deprecated due to API redundancy. Use analyze() instead. "
            "fit_signal() returns (fit_func, FitResult) but FitResult.fit_func already "
            "contains the same function.",
            DeprecationWarning,
            stacklevel=2
        )
    
    fit_result = _fit_signal_core(
        freq, signal, geometry, fit_amplitude, fit_edelay, 
        final_ls_opti, allow_mismatch, rec_depth, suppress_warnings
    )
    
    return fit_result.fit_func, fit_result

def analyze(
    freq: np.ndarray,
    signal: np.ndarray,
    geometry: str,
    fit_amplitude: bool = True,
    fit_edelay: bool = True,
    final_ls_opti: bool = True,
    allow_mismatch: bool = True,
    rec_depth: int = 1,
) -> FitResult:
    """Analyze resonator S-parameter data to extract physical parameters.

    This is the recommended function for resonator parameter extraction. It returns
    a comprehensive FitResult object containing fitted parameters, uncertainties,
    and quality assessment tools.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array in Hz.
    signal : np.ndarray
        Complex S-parameter data to fit.
    geometry : str
        Resonator measurement geometry:
        - "r", "reflection": Reflection measurement (S₁₁)
        - "rm", "reflection_mismatched": Reflection with mismatch
        - "t", "transmission": Transmission measurement (S₂₁)
        - "h", "hanger": Hanger coupling measurement
        - "hm", "hanger_mismatched": Hanger with mismatch
    fit_amplitude : bool, optional
        If True, fit complex amplitude scaling (default: True).
    fit_edelay : bool, optional
        If True, estimate and fit electrical delay (default: True).
    final_ls_opti : bool, optional
        If True, perform final nonlinear optimization (default: True).
    allow_mismatch : bool, optional
        If True, automatically use mismatched models when appropriate (default: True).
    rec_depth : int, optional
        ABCD algorithm recursion depth, 0-2 recommended (default: 1).

    Returns
    -------
    FitResult
        Comprehensive fit result object containing:
        - Fitted parameters with uncertainty estimates
        - Goodness of fit metrics (R², reduced χ², RMS residual)
        - Parameter correlation matrix
        - Fit validation with warnings and recommendations
        - Save/load functionality for analysis results
        - Direct access to fit function via .fit_func attribute

    Examples
    --------
    >>> # Basic fitting
    >>> result = analyze(freq, s21_data, "transmission")
    >>> print(f"f₀ = {result.f_0:.6e} Hz")
    >>> print(f"Q = {result.f_0/result.kappa:.1f}")
    >>> 
    >>> # Access uncertainties
    >>> f0_error = result.get_param_error('f_0')
    >>> print(f"f₀ = {result.f_0:.6e} ± {f0_error:.2e} Hz")
    >>> 
    >>> # Evaluate fit function
    >>> fitted_signal = result.fit_func(freq, *result.params.params)
    >>> 
    >>> # Validate fit quality
    >>> validation = result.validate_fit(freq, s21_data)
    >>> print(f"Fit status: {validation['status']}")
    >>> 
    >>> # Save results
    >>> result.save_to_file('my_fit_results.json')

    Notes
    -----
    This function provides a clean API without redundancy. The fitted function
    is available as result.fit_func, eliminating the need to return it separately
    as in the deprecated fit_signal() function.

    For plotting results, use the .plot() method:
    >>> result.plot(freq, signal)
    """
    return _fit_signal_core(
        freq, signal, geometry, fit_amplitude, fit_edelay,
        final_ls_opti, allow_mismatch, rec_depth, suppress_warnings=False
    )
