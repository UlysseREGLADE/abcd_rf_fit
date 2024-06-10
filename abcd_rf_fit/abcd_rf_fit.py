from inspect import signature
import numpy as np
import warnings

from .utils import (
    guess_edelay_from_gradient,
    smooth_gradient,
    complex_fit,
)

from .resonators import *

from .fraction_fit import get_rationnal_fit

import matplotlib.pyplot as plt

def get_abcd(freq, signal, rec_depth=0):

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


def abcd2params(abcd, geometry):

    if len(abcd) == 4:
        a, b, c, d = abcd
    else:
        a, b, c, d, e, f = abcd

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

        phi_0 = np.angle(kappa_c_real - 1j*kappa_c_imag)

        return f_0, kappa, kappa_c_real, phi_0, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == transmission:

        signal_f_0_before = (a - b * np.real(c / d)) / (c - d * np.real(c / d))
        a, b = a - c * b / d, 0
        signal_f_0_after = (a - b * np.real(c / d)) / (c - d * np.real(c / d))
        a = a * signal_f_0_before / signal_f_0_after

        kappa = -2 * np.imag(c / d)
        f_0 = -np.real(c / d)
        a_in = 2j * a / d

        return f_0, kappa, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == hanger:

        kappa_c_r = np.real(1j * (c / d - a / b))
        kappa_i_r = -np.imag(a / b + c / d)
        f_0 = -0.5 * np.real(a / b + c / d)
        a_in = b / d

        kappa_c_real = 2 * kappa_c_r
        kappa_i = kappa_i_r - kappa_c_r

        kappa = kappa_i + kappa_c_real

        return f_0, kappa, kappa_c_real, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == hanger_mismatched:

        f_0 = -np.real(c / d)
        a_in = b / d

        kappa_c_imag = 2*np.real(a / b - c / d)
        kappa_c_real = 2*np.real(1j * (c / d - a / b))
        kappa_i = -2 * np.imag(c / d) - kappa_c_real

        kappa = kappa_i + kappa_c_real

        phi_0 = np.angle(kappa_c_real - 1j*kappa_c_imag)

        return f_0, kappa, kappa_c_real, phi_0, np.real(a_in), np.imag(a_in)
    
    elif resonator_dict[geometry] == purcell_reflection:

        kappa_a = -np.imag(e / f + b / c)
        kappa_b = -np.imag(e / f - b / c)
        f_a_0 = -np.imag(a / c - d / f) / kappa_b
        f_b_0 = +np.imag(a / c + d / f) / kappa_a
        a_in = c / f

        # The formula below is true, but since g**2 is much smaller than f_a_0 * f_b_0
        # in practice it dose not work
        # g = np.abs(np.real(-0.5 * a / c - 0.5 * d / f + f_a_0 * f_b_0)) ** 0.5

        # This is why we compute g from the signal at f_a_0
        num_minus_g2 = (1j * (f_a_0-f_b_0) - 0.5 * kappa_b) * (0.5 * kappa_a)
        den_minus_g2 = (1j * (f_a_0-f_b_0) + 0.5 * kappa_b) * (0.5 * kappa_a)
        signal_at_f_a_0 = (a+b*f_a_0+c*f_a_0**2)/(d+e*f_a_0+f*f_a_0**2)
        g = np.abs((a_in*num_minus_g2-signal_at_f_a_0*den_minus_g2)/(a_in-signal_at_f_a_0))**0.5

        return f_a_0, f_b_0, kappa_a, kappa_b, g, np.real(a_in), np.imag(a_in)


def get_fit_function(geometry, amplitude=True, edelay=True):
    resonator_func = resonator_dict[geometry]

    if not amplitude and not edelay:

        return resonator_func

    elif amplitude and not edelay:

        def fit_func(*args):
            return resonator_func(*args[:-2]) * (args[-2] + 1j * args[-1])

        return fit_func

    elif not amplitude and edelay:

        def fit_func(*args):
            return resonator_func(*args[:-1]) * np.exp(2j * np.pi * args[-1] * args[0])

        return fit_func

    elif amplitude and edelay:

        def fit_func(*args):
            return (
                resonator_func(*args[:-3])
                * (args[-3] + 1j * args[-2])
                * np.exp(2j * np.pi * args[-1] * args[0])
            )

        return fit_func

    else:

        raise Exception("Unreachable")


def meta_fit_edelay(freq, signal, relative_edelay_span, quick_fit=None):
    if quick_fit is None:
        def quick_fit(freq, signal):
            return get_abcd(freq, signal, rec_depth=2)

    guess_edelay = guess_edelay_from_gradient(freq, signal)

    edelay_span = relative_edelay_span / (np.max(freq) - np.min(freq))

    edelay_array = guess_edelay + edelay_span*np.linspace(-1, 1, int(24*relative_edelay_span))
    l2_error_array = np.zeros_like(edelay_array)

    for i, ed in enumerate(edelay_array):

        s = signal * np.exp(-2j * np.pi * freq * ed)
        _, abcd_fit = quick_fit(freq, s)

        l2_error_array[i] = np.sum(np.abs(s - abcd_fit) ** 2) / freq.size

    return edelay_array[np.argmin(l2_error_array)]


def fit_signal(
    freq,
    signal,
    geometry,
    fit_amplitude=True,
    fit_edelay=True,
    final_ls_opti=True,
    allow_mismatch=True,
    return_abcd=False,
):
    
    if resonator_dict[geometry] == purcell_reflection:
        def quick_fit(freq, signal):
            return get_rationnal_fit(freq, signal, n=2, return_true_solution=False, min_rec_depth=6, max_rec_depth=12, min_converged_passes=3)
        def true_fit(freq, signal):
            return get_rationnal_fit(freq, signal, n=2)
        
        relative_edelay_span = 4.5
    else:
        def quick_fit(freq, signal):
            return get_abcd(freq, signal, rec_depth=1)
        def true_fit(freq, signal):
            return get_abcd(freq, signal, rec_depth=2)
        
        relative_edelay_span = 2.5

    if fit_edelay:
        edelay = meta_fit_edelay(freq, signal, relative_edelay_span, quick_fit)
    else:
        edelay = 0

    if resonator_dict[geometry] == reflection and allow_mismatch:
        geometry = "rm"
    elif resonator_dict[geometry] == hanger and allow_mismatch:
        geometry = "hm"

    corrected_signal = signal * np.exp(-2j * np.pi * edelay * freq)

    abcd, _ = true_fit(freq, corrected_signal)

    params = abcd2params(abcd, geometry)
    if not fit_amplitude:
        params = params[:-2]
    if fit_edelay:
        params = [*params, edelay]
    params = np.array(params)

    fit_func = get_fit_function(geometry, fit_amplitude, fit_edelay)

    if final_ls_opti:
        # TODO : correct this line, there might be a division by zero here
        params_scaling = np.copy(params)

        def scaled_fit_func(freq, *normed_params):
            args = np.array(normed_params)*params_scaling
            return fit_func(freq, *args)
        
        normed_params, _ = complex_fit(scaled_fit_func, freq, signal, params/params_scaling)
        params = normed_params*params_scaling

    
    resonator_params = ResonatorParams(params, geometry)

    if resonator_params.phi_0 is not None and  np.abs(resonator_params.phi_0) > 0.25:
        warnings.warn("Extracted phi_0 greater than 0.25, this might indicate a big impedance mismatch, values of kappa_i and kappa_c might be affected, you can try to set: allow_mismatch=False", UserWarning)

    if return_abcd:
        return fit_func, ResonatorParams(params, geometry), abcd
    return fit_func, ResonatorParams(params, geometry)
