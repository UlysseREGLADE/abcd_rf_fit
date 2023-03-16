from inspect import signature
import numpy as np

from .utils import (
    guess_edelay_from_gradient,
    smooth_gradient,
    complex_fit,
)

from .resonators import *


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

    a, b, c, d = abcd

    if resonator_dict[geometry] == reflection:

        kappa_c = np.real(1j * (c / d - a / b))
        kappa_i = -np.imag(a / b + c / d)
        f_0 = -0.5 * np.real(a / b + c / d)
        a_in = b / d

        return f_0, kappa_i, kappa_c, np.real(a_in), np.imag(a_in)

    if resonator_dict[geometry] == reflection_mismatched:

        kappa_c_h = 2 * np.abs(a / b - c / d)
        phi_0 = np.angle(1j * (c / d - a / b))
        kappa_i_h = -2 * np.imag(c / d) - kappa_c_h
        f_0 = -np.real(c / d)
        a_in = b / d

        kappa_c = kappa_c_h / 2
        kappa_i = kappa_i_h + kappa_c_h / 2

        return f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in)

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

        kappa_c = 2 * kappa_c_r
        kappa_i = kappa_i_r - kappa_c_r

        return f_0, kappa_i, kappa_c, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == hanger_mismatched:

        kappa_c = 2 * np.abs(a / b - c / d)
        phi_0 = np.angle(1j * (c / d - a / b))
        kappa_i = -2 * np.imag(c / d) - kappa_c
        f_0 = -np.real(c / d)
        a_in = b / d

        return f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in)


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


def meta_fit_edelay(freq, signal, rec_depth=0):

    quick_fit = get_abcd

    guess_edelay = guess_edelay_from_gradient(freq, signal)

    edelay_span = 1.5 / (np.max(freq) - np.min(freq))

    edelay_array = guess_edelay + np.linspace(-1, 1, 1001) * edelay_span
    l2_error_array = np.zeros_like(edelay_array)

    for i, ed in enumerate(edelay_array):

        s = signal * np.exp(-2j * np.pi * freq * ed)
        _, abcd_fit = quick_fit(freq, s, rec_depth)

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
    rec_depth=1,
):

    if fit_edelay:
        edelay = meta_fit_edelay(freq, signal, rec_depth)
    else:
        edelay = 0

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

    if final_ls_opti:

        params, _ = complex_fit(fit_func, freq, signal, params)

    return fit_func, ResonatorParams(params, geometry)