import numpy as np
from .abcd_rf_fit import *


def reflection_purcell(freq, f_a_0, f_b_0, kappa_a, kappa_b, g):

    delta_a = freq - f_a_0
    delta_b = freq - f_b_0

    num = (1j * delta_b - 0.5 * kappa_b) * (1j * delta_a + 0.5 * kappa_a) + g ** 2
    den = (1j * delta_b + 0.5 * kappa_b) * (1j * delta_a + 0.5 * kappa_a) + g ** 2

    return num / den


def hanger_purcell(freq, f_a_0, f_b_0, kappa_a, kappa_b, g):

    delta_a = freq - f_a_0
    delta_b = freq - f_b_0

    num = (1j * delta_b) * (1j * delta_a + 0.5 * kappa_a) + g ** 2
    den = (1j * delta_b + 0.5 * kappa_b) * (1j * delta_a + 0.5 * kappa_a) + g ** 2

    return num / den


resonator_dict = {
    "transmission": transmission,
    "t": transmission,
    "reflection": reflection,
    "r": reflection,
    "reflection_mismatched": reflection_mismatched,
    "rm": reflection_mismatched,
    "hanger": hanger,
    "h": hanger,
    "hanger_mismatched": hanger_mismatched,
    "hm": hanger_mismatched,
    "rp": reflection_purcell,
    "hp": hanger_purcell,
}


def get_abcdef(freq, signal, _=None):

    freq_center = np.mean(freq)

    x_design = np.ones((3, freq.size))
    x_design[1, :] = freq - freq_center
    x_design[2, :] = (freq - freq_center) ** 2

    signal_grad = smooth_gradient(signal)

    xx = (x_design * np.abs(signal_grad)) @ x_design.T
    xdyx = (signal * x_design * np.abs(signal_grad)) @ x_design.T
    xdycx = (np.conj(signal) * x_design * np.abs(signal_grad)) @ x_design.T
    xdy2x = (np.abs(signal) ** 2 * x_design * np.abs(signal_grad)) @ x_design.T

    up_right = np.linalg.inv(xx) @ xdyx
    bottom_left = np.linalg.inv(xdy2x) @ xdycx

    to_diag = np.zeros((6, 6), dtype=complex)
    to_diag[:3, 3:] = up_right
    to_diag[3:, :3] = bottom_left

    v, w = np.linalg.eig(to_diag)

    abcdef = w[:, np.argmin(np.abs(1 - v))]

    abcdef[0::3] += abcdef[2::3] * freq_center ** 2 - abcdef[1::3] * freq_center
    abcdef[1::3] -= 2 * abcdef[2::3] * freq_center

    fit = (abcdef[0] + abcdef[1] * freq + abcdef[2] * freq ** 2) / (
        abcdef[3] + abcdef[4] * freq + abcdef[5] * freq ** 2
    )

    return abcdef, fit


def meta_fit_edelay(freq, signal, rec_depth=0, max_turns=1):

    quick_fit = get_abcdef

    guess_edelay = guess_edelay_from_gradient(freq, signal)

    edelay_span = 2.5 / (np.max(freq) - np.min(freq))

    edelay_array = guess_edelay + np.linspace(-1, 1, 1001) * edelay_span
    l2_error_array = np.zeros_like(edelay_array)

    for i, ed in enumerate(edelay_array):

        s = signal * np.exp(-2j * np.pi * freq * ed)
        _, abcd_fit = quick_fit(freq, s, rec_depth)

        l2_error_array[i] = np.sum(np.abs(s - abcd_fit) ** 2) / freq.size

    return edelay_array[np.argmin(l2_error_array)]


def abcd2params(abcd, geometry):

    if resonator_dict[geometry] in [reflection, transmission, hanger]:
        a, b, c, d = abcd
    elif resonator_dict[geometry] in [reflection_purcell, hanger_purcell]:
        a, b, c, d, e, f = abcd

    if resonator_dict[geometry] == reflection:

        kappa_c_h = 2 * np.abs(a / b - c / d)
        phi_0 = np.angle(1j * (c / d - a / b))
        kappa_i_h = -2 * np.imag(c / d) - kappa_c_h
        f_0 = -0.5 * np.real(a / b + c / d)
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
        a_in = 2j * a / (d * kappa)

        return f_0, kappa, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == hanger:

        kappa_c = 2 * np.abs(a / b - c / d)
        phi_0 = np.angle(1j * (c / d - a / b))
        kappa_i = -2 * np.imag(c / d) - kappa_c
        f_0 = -0.5 * np.real(a / b + c / d)
        a_in = b / d

        return f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == reflection_purcell:

        kappa_a = -np.imag(e / f + b / c)
        kappa_b = -np.imag(e / f - b / c)
        f_a_0 = -np.imag(a / c - d / f) / kappa_b
        f_b_0 = +np.imag(a / c + d / f) / kappa_a
        g = np.abs(np.real(-0.5 * a / c - 0.5 * d / f + f_a_0 * f_b_0)) ** 0.5
        a_in = c / f

        return f_a_0, f_b_0, kappa_a, kappa_b, g, np.real(a_in), np.imag(a_in)
