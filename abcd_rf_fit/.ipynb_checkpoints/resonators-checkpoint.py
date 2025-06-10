import numpy as np

if __name__ == "__main__":

    from utils import (
        get_prefix_str,
        zeros2eps,
    )

else:

    from .utils import (
        get_prefix_str,
        zeros2eps,
    )


def transmission(freq, f_0, kappa):

    num = 1
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection(freq, f_0, kappa_i, kappa_c, phi_0=0):

    num = 1j * (freq - f_0) + kappa_i / 2 + kappa_c * (0.5 - np.exp(1j * phi_0))
    den = 1j * (freq - f_0) + (kappa_c + kappa_i) / 2

    return num / zeros2eps(den)


def reflection_mismatched(freq, f_0, kappa_i, kappa_c, phi_0):

    return reflection(freq, f_0, kappa_i, kappa_c, phi_0)


def hanger(freq, f_0, kappa_i, kappa_c, phi_0=0):

    num = 1j * (freq - f_0) + kappa_i / 2 + kappa_c / 2 * (1 - np.exp(1j * phi_0))
    den = 1j * (freq - f_0) + (kappa_c + kappa_i) / 2

    return num / zeros2eps(den)


def hanger_mismatched(freq, f_0, kappa_i, kappa_c, phi_0):

    return hanger(freq, f_0, kappa_i, kappa_c, phi_0)


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


class ResonatorParams:
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
            self.kappa_i_index = 1
            self.kappa_c_index = 2
            if len(self.params) in [5, 6]:
                self.re_a_in_index = 3
                self.im_a_in_index = 4
            if len(self.params) in [4, 6]:
                self.edelay_index = -1

        if self.resonator_func in [reflection_mismatched, hanger_mismatched]:
            self.f_0_index = 0
            self.kappa_i_index = 1
            self.kappa_c_index = 2
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
        return None

    @property
    def kappa(self):
        if hasattr(self, "kappa_index"):
            return self.params[self.kappa_index]
        return self.params[self.kappa_i_index] + self.params[self.kappa_c_index]

    @property
    def kappa_i(self):
        if hasattr(self, "kappa_i_index"):
            return self.params[self.kappa_i_index]
        return None

    @property
    def kappa_c(self):
        if hasattr(self, "kappa_c_index"):
            return self.params[self.kappa_c_index]
        return None

    @property
    def a_in(self):
        if hasattr(self, "re_a_in_index") and hasattr(self, "im_a_in_index"):
            return (
                self.params[self.re_a_in_index] + 1j * self.params[self.im_a_in_index]
            )
        return None

    @property
    def re_a_in(self):
        a_in = self.a_in
        if a_in is not None:
            return np.real(a_in)
        return None

    @property
    def im_a_in(self):
        a_in = self.a_in
        if a_in is not None:
            return np.imag(a_in)
        return None

    @property
    def edelay(self):
        if hasattr(self, "edelay_index"):
            return self.params[self.edelay_index]
        return None

    @property
    def phi_0(self):
        if hasattr(self, "phi_0_index"):
            return self.params[self.phi_0_index]
        return None

    def str(self, latex=False, separator=", ", precision=2, only_f_and_kappa=False, f_precision=2):

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
