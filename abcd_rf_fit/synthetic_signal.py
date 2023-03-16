from sqlite3 import paramstyle
import numpy as np
from scipy.interpolate import interp1d
from .abcd_rf_fit import *


def get_background(freq, non_uniform_background=False):

    n_indexs = 5
    indexs = np.random.randint(1, freq.size - 1, size=(n_indexs))
    while len(set(indexs)) < 5:
        indexs = np.random.randint(1, freq.size - 1, size=(n_indexs))
    indexs = np.sort(indexs)
    indexs[0], indexs[-1] = 0, freq.size - 1

    base_edelay = np.random.rand() * 1000e-9 + 5e-9
    # var_edelay = (np.random.rand(n_indexs)) * 5e-9
    # edelay_interp = interp1d(freq[indexs], base_edelay + var_edelay * 0, kind="cubic")

    background = np.exp(2j * np.pi * base_edelay * freq)
    if non_uniform_background:
        base_amplitude = 0.9
        var_amplitude = (np.random.rand(n_indexs) - 0.5) * 0.1
        amplitude_interp = interp1d(
            freq[indexs], base_amplitude + var_amplitude, kind="cubic"
        )
        background *= amplitude_interp(freq)

    return background, base_edelay


def get_synthetic_resonance(freq, geometry="r"):

    freq_span = freq[-1] - freq[0]
    kappa_i = (0.001 + 0.29 * np.random.rand()) * freq_span
    kappa_c = (0.1 + 2 * np.random.rand()) * kappa_i
    f_0 = (0.25 + np.random.rand() * 0.5) * freq_span + freq[0]
    phi_0 = (np.random.rand() - 0.5) * np.pi / 2

    resonator_func = resonator_dict[geometry]

    if resonator_func == transmission:
        params = [f_0, (kappa_i + kappa_c) * 0.5]
    elif resonator_func == reflection:
        params = [f_0, kappa_i, kappa_c]
    elif resonator_func == hanger:
        params = [f_0, kappa_i, kappa_c]
    elif resonator_func == reflection_mismatched:
        params = [f_0, kappa_i, kappa_c, phi_0]
    elif resonator_func == hanger_mismatched:
        params = [f_0, kappa_i, kappa_c, phi_0]

    return resonator_func(freq, *params), params


def get_synthetic_signal(geometry="r"):

    freq_center = np.random.rand() * 6e9 + 2e9
    freq_span = np.random.rand() * 10e6 + 100e3
    n_freq = np.random.randint(201, 2001)

    freq = np.linspace(freq_center - freq_span / 2, freq_center + freq_span / 2, n_freq)

    background, edelay = get_background(freq)
    resonance, params = get_synthetic_resonance(freq, geometry)

    params = ResonatorParams([*params, edelay], geometry)

    signal = background * resonance
    white_noise = np.random.normal(0, 1, size=freq.size) + 1j * np.random.normal(
        0, 1, size=freq.size
    )
    noise = 0.2 * np.mean(np.abs(signal)) * np.random.rand() * white_noise

    return freq, signal + noise, params

