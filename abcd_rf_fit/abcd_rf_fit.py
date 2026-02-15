import warnings

import numpy as np

from .resonators import (
    ResonatorParams,
    ResonatorCollection,
    get_fit_function,
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
from .vector_fit import vector_fit_siso


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


def meta_fit_edelay(
    freq, 
    signal,
    rec_depth=0,
    fit_method: str = 'abcd',
    n_poles:int = 1,
    n_guess: int = 1001,
):
    assert fit_method in ['abcd', 'vector_fit']

    if fit_method == 'abcd':
        def quick_fit(freq, s, rec_depth):
            abcd_fit = get_abcd(freq, s, rec_depth)[1]
            return np.sum(np.abs(s - abcd_fit) ** 2) / freq.size
    else:
        def quick_fit(freq, s, rec_depth):
            return vector_fit_siso(freq, s, n_poles, rec_depth)['least_square_error']

    guess_edelay = guess_edelay_from_gradient(freq, signal)

    edelay_span = 1.5 / (np.max(freq) - np.min(freq))

    edelay_array = guess_edelay + np.linspace(-1, 1, n_guess) * edelay_span
    l2_error_array = np.zeros_like(edelay_array)

    for i, ed in enumerate(edelay_array):
        s = signal * np.exp(-2j * np.pi * freq * ed)
        l2_error_array[i] = quick_fit(freq, s, rec_depth)

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
    api_warning=True,
):
    if api_warning:
        warnings.warn("fit_signal() is deprecated, please use analyze() instead, and analyze().plot() to display data.", UserWarning)

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
    
    resonator_params = ResonatorParams(params, geometry, freq, signal)

    if resonator_params.phi_0 is not None and  np.abs(resonator_params.phi_0) > 0.25:
        warnings.warn("Extracted phi_0 greater than 0.25, this might indicate a big impedance mismatch, values of kappa_i and kappa_c might be affected, you can try to set: allow_mismatch=False", UserWarning)

    return fit_func, ResonatorParams(params, geometry, freq, signal)

def analyze(
    freq: np.ndarray,
    signal: np.ndarray,
    geometry: str | list[str] = 'r',
    fit_amplitude: bool = True,
    fit_edelay: bool =True,
    final_ls_opti: bool =True,
    allow_mismatch: bool =True,
    rec_depth: int | None = None,
):
    if isinstance(geometry, str):
        rec_depth = 1
        return fit_signal(
            freq,
            signal,
            geometry,
            fit_amplitude,
            fit_edelay,
            final_ls_opti,
            allow_mismatch,
            rec_depth,
            api_warning = False)[1]
    else:
        rec_depth = 50
        for g in geometry:
            assert resonator_dict[g] is not transmission

        if fit_edelay:
            edelay = meta_fit_edelay(
                freq,
                signal,
                rec_depth=rec_depth,
                fit_method='vector_fit',
                n_guess = 101
            )
        else:
            edelay = 0

        corrected_signal = signal * np.exp(-2j * np.pi * edelay * freq)
        vector_fit_dict = vector_fit_siso(
            freq,
            corrected_signal,
            n_poles=len(geometry),
            n_iter=rec_depth
        )

        resonators = []
        for i, g in enumerate(geometry):
            resonator_geometry = g
            params = [
                vector_fit_dict['poles'][i].real,
                2*vector_fit_dict['poles'][i].imag
            ]
            if resonator_dict[g] is reflection:
                params.append(
                    np.imag(vector_fit_dict['residues'][i]/vector_fit_dict['r0'])
                )
            elif resonator_dict[g] is hanger:
                params.append(
                    2*np.imag(vector_fit_dict['residues'][i]/vector_fit_dict['r0'])
                )
            if allow_mismatch:
                params.append(
                    np.angle(-1j*vector_fit_dict['residues'][i]/vector_fit_dict['r0'])
                )
                if resonator_dict[g] is reflection:
                    resonator_geometry = "rm"
                elif resonator_dict[g] is hanger:
                    resonator_geometry = "hm"
            params.append(np.real(vector_fit_dict['r0']))
            params.append(np.imag(vector_fit_dict['r0']))
            params.append(edelay)

            resonator = ResonatorParams(
                params,
                resonator_geometry,
                freq,
                signal,
            )
            resonators.append(resonator)
        
        # TODO: final least square
        return ResonatorCollection(
            resonators=resonators,
            a_in=vector_fit_dict['r0'],
            edelay=edelay,
            signal=signal,
            freq=freq
        )