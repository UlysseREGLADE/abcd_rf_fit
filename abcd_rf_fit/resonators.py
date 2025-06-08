from inspect import signature
import numpy as np
from .utils import (
    guess_edelay_from_gradient,
    smooth_gradient,
    complex_fit,
)

from .resonators import *


class FitResult(object):
    """
    Container for fit results including parameters, covariance matrix, and quality metrics
    """

    def __init__(self, params, geometry, pcov=None, fit_func=None):
        self.resonator_params = ResonatorParams(params, geometry)
        self.pcov = pcov
        self.fit_func = fit_func

    @property
    def params(self):
        """Access to the ResonatorParams object"""
        return self.resonator_params

    @property
    def param_errors(self):
        """Calculate parameter uncertainties from covariance matrix"""
        if self.pcov is not None:
            return np.sqrt(np.diag(self.pcov))
        else:
            return None

    @property
    def correlation_matrix(self):
        """Calculate correlation matrix from covariance matrix"""
        if self.pcov is not None:
            std_devs = np.sqrt(np.diag(self.pcov))
            correlation = self.pcov / np.outer(std_devs, std_devs)
            return correlation
        else:
            return None

    def get_param_error(self, param_name):
        """Get uncertainty for a specific parameter"""
        if self.param_errors is None:
            return None

        # Map parameter names to indices
        param_map = {}
        if hasattr(self.resonator_params, "f_0_index"):
            param_map["f_0"] = self.resonator_params.f_0_index
        if hasattr(self.resonator_params, "kappa_index"):
            param_map["kappa"] = self.resonator_params.kappa_index
        if hasattr(self.resonator_params, "kappa_c_real_index"):
            param_map["kappa_c"] = self.resonator_params.kappa_c_real_index
        if hasattr(self.resonator_params, "phi_0_index"):
            param_map["phi_0"] = self.resonator_params.phi_0_index
        if hasattr(self.resonator_params, "re_a_in_index"):
            param_map["re_a_in"] = self.resonator_params.re_a_in_index
        if hasattr(self.resonator_params, "im_a_in_index"):
            param_map["im_a_in"] = self.resonator_params.im_a_in_index
        if hasattr(self.resonator_params, "edelay_index"):
            param_map["edelay"] = self.resonator_params.edelay_index

        if param_name in param_map:
            return self.param_errors[param_map[param_name]]
        else:
            return None

    def goodness_of_fit(self, freq, signal):
        """Calculate goodness of fit metrics"""
        if self.fit_func is None:
            return None

        fitted_signal = self.fit_func(freq, *self.resonator_params.params)
        residuals = signal - fitted_signal

        # R-squared
        ss_res = np.sum(np.abs(residuals) ** 2)
        ss_tot = np.sum(np.abs(signal - np.mean(signal)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Reduced chi-squared
        n_params = len(self.resonator_params.params)
        n_data = len(signal)
        reduced_chi_sq = ss_res / (n_data - n_params)

        return {
            "r_squared": r_squared,
            "reduced_chi_squared": reduced_chi_sq,
            "rms_residual": np.sqrt(ss_res / n_data),
        }

    def __str__(self):
        result = str(self.resonator_params)
        if self.param_errors is not None:
            result += "\nParameter uncertainties available (use .param_errors or .get_param_error())"
        return result

    def __repr__(self):
        return self.__str__()




if __name__ == "__main__":

    from utils import (
        zeros2eps,
        get_prefix_str,
    )

else:

    from .utils import (
        zeros2eps,
        get_prefix_str,
    )


def transmission(freq, f_0, kappa):

    num = 1
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection(freq, f_0, kappa, kappa_c_real, phi_0=0):

    num = 2j * (freq - f_0) + kappa - 2*kappa_c_real*(1+1j*np.tan(phi_0))
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection_mismatched(freq, f_0, kappa, kappa_c_real, phi_0):

    return reflection(freq, f_0, kappa, kappa_c_real, phi_0)


def hanger(freq, f_0, kappa, kappa_c_real, phi_0=0):

    num = 2j * (freq - f_0) + kappa - kappa_c_real*(1+1j*np.tan(phi_0))
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def hanger_mismatched(freq, f_0, kappa, kappa_c_real, phi_0):

    return hanger(freq, f_0, kappa, kappa_c_real, phi_0)


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
}


class ResonatorParams(object):
    def __init__(self, params, geometry):

        self.resonator_func = resonator_dict[geometry]
        self.params = params

        if self.resonator_func == transmission:
            self.f_0_index = 0
            self.kappa_index = 1
            if len(self.params) in [4, 5]:
                self.re_a_in_index = 2
                self.im_a_in_index = 3
            if len(self.params) in [3, 5]:
                self.edelay_index = -1

        if self.resonator_func in [reflection, hanger]:
            self.f_0_index = 0
            self.kappa_index = 1
            self.kappa_c_real_index = 2
            if len(self.params) in [5, 6]:
                self.re_a_in_index = 3
                self.im_a_in_index = 4
            if len(self.params) in [4, 6]:
                self.edelay_index = -1

        if self.resonator_func in [reflection_mismatched, hanger_mismatched]:
            self.f_0_index = 0
            self.kappa_index = 1
            self.kappa_c_real_index = 2
            self.phi_0_index = 3
            if len(self.params) in [6, 7]:
                self.re_a_in_index = 3
                self.im_a_in_index = 4
            if len(self.params) in [5, 7]:
                self.edelay_index = -1

    def tolist(self):
        return np.array(self.params)

    @property
    def f_0(self):
        if hasattr(self, "f_0_index"):
            return self.params[self.f_0_index]
        else:
            return None

    @property
    def kappa(self):
        if hasattr(self, "kappa_index"):
            return self.params[self.kappa_index]
        else:
            None

    @property
    def kappa_i(self):
        if hasattr(self, "kappa_index") and hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_index] - self.params[self.kappa_c_real_index]
        else:
            return None

    @property
    def kappa_c_real(self):
        if hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_c_real_index]
        else:
            return None

    @property
    def kappa_c(self):
        return self.kappa_c_real

    @property
    def a_in(self):
        if hasattr(self, "re_a_in_index") and hasattr(self, "im_a_in_index"):
            return (
                self.params[self.re_a_in_index] + 1j * self.params[self.im_a_in_index]
            )
        else:
            return None

    @property
    def re_a_in(self):
        a_in = self.a_in
        if a_in is not None:
            return np.real(a_in)
        else:
            return None

    @property
    def im_a_in(self):
        a_in = self.a_in
        if a_in is not None:
            return np.imag(a_in)
        else:
            return None

    @property
    def edelay(self):
        if hasattr(self, "edelay_index"):
            return self.params[self.edelay_index]
        else:
            return None

    @property
    def phi_0(self):
        if hasattr(self, "phi_0_index"):
            return self.params[self.phi_0_index]
        else:
            return None

    def str(self, latex=False, separator=", ", precision=2, only_f_and_kappa=False, f_precision=5):

        kappa = {False: "kappa/2pi", True: r"$\kappa/2\pi$"}
        kappa_i = {False: "kappa_i/2pi", True: r"$\kappa_i/2\pi$"}
        kappa_c = {False: "kappa_c/2pi", True: r"$\kappa_c/2\pi$"}
        phi_0 = {False: "phi_0", True: r"$\varphi_0$"}
        f_0 = {False: "f_0", True: r"$f_0$"}

        if self.edelay is not None:
            edelay_str = "%sedelay = %ss" % (separator, get_prefix_str(self.edelay, precision))
        else:
            edelay_str = ""

        if self.resonator_func == transmission:
            kappa_str = r"%s%s = %sHz" % (
                separator,
                kappa[latex],
                get_prefix_str(self.kappa, precision),
            )
        else:
            kappa_str = r"%s%s = %sHz%s%s = %sHz" % (
                separator,
                kappa_i[latex],
                get_prefix_str(self.kappa_i, precision),
                separator,
                kappa_c[latex],
                get_prefix_str(self.kappa_c, precision),
            )

        if self.resonator_func in [hanger_mismatched, reflection_mismatched]:
            phi_0_str = r"%s%s = %0.2f rad" % (separator, phi_0[latex], self.phi_0)
        else:
            phi_0_str = ""

        f_0_str = r"%s = %sHz" % (f_0[latex], get_prefix_str(self.f_0, f_precision))

        if only_f_and_kappa:
            return f_0_str + kappa_str
        else:
            return f_0_str + kappa_str + phi_0_str + edelay_str

    def __str__(self) -> str:

        return self.str()

    def __repr__(self):

        return self.str()


if __name__ == "__main__":

    f_0 = 3.8e9
    kappa_i = 50e6
    kappa_c = 150e6
    a_in = 1 + 1j
    edelay = 32e-9
    phi_0 = 4
    geometry = "hm"

    params = [f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in), edelay]

    params = ResonatorParams(params, geometry)

    print(params)

    f_0 = 3.8e9
    kappa_i = 50e6
    kappa_c = 150e6
    a_in = 1 + 1j
    edelay = 32e-9
    geometry = "r"

    params = [f_0, kappa_i, kappa_c, np.real(a_in), np.imag(a_in), edelay]

    params = ResonatorParams(params, geometry)

    print(params)

    f_0 = 3.8e9
    kappa = 50e6
    a_in = 1 + 1j
    geometry = "t"

    params = [f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in)]

    params = ResonatorParams(params, geometry)

    print(params)
