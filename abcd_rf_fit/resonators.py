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

    def validate_fit(self, freq, signal, strict=False):
        """
        Comprehensive fit validation with automatic warnings and recommendations

        Args:
            freq: Frequency array
            signal: Signal array
            strict: If True, apply stricter validation criteria

        Returns:
            dict: Validation results with status, warnings, and recommendations
        """
        validation = {
            'status': 'good',
            'warnings': [],
            'recommendations': [],
            'metrics': {}
        }

        # 1. Parameter uncertainty validation
        if self.param_errors is not None:
            # Check if any parameter has very large uncertainty
            rel_errors = []
            param_names = ['f_0', 'kappa_c', 'edelay']

            for param_name in param_names:
                param_val = getattr(self.resonator_params, param_name, None)
                param_err = self.get_param_error(param_name)

                if param_val is not None and param_err is not None and param_val != 0:
                    rel_error = abs(param_err / param_val)
                    rel_errors.append((param_name, rel_error))
                    validation['metrics'][f'{param_name}_relative_error'] = rel_error

                    # Flag parameters with high uncertainty
                    threshold = 0.1 if strict else 0.2  # 10% or 20% relative error
                    if rel_error > threshold:
                        validation['warnings'].append(
                            f"{param_name} has high uncertainty ({rel_error*100:.1f}%)"
                        )
                        validation['recommendations'].append(
                            f"Consider more data points or lower noise for better {param_name} determination"
                        )

        # 2. Goodness of fit validation
        gof = self.goodness_of_fit(freq, signal)
        if gof is not None:
            validation['metrics'].update(gof)

            # R-squared validation
            r2_threshold = 0.95 if strict else 0.9
            if gof['r_squared'] < r2_threshold:
                validation['warnings'].append(
                    f"Low R² = {gof['r_squared']:.3f} (< {r2_threshold})"
                )
                validation['recommendations'].append(
                    "Consider different geometry or check for systematic errors"
                )

            # Reduced chi-squared validation
            if gof['reduced_chi_squared'] > 5:
                validation['warnings'].append(
                    f"High χ²_red = {gof['reduced_chi_squared']:.2f} (>> 1)"
                )
                validation['recommendations'].append(
                    "Systematic residuals detected - model may be inadequate"
                )
            elif gof['reduced_chi_squared'] < 0.1:
                validation['warnings'].append(
                    f"Very low χ²_red = {gof['reduced_chi_squared']:.2f} (<< 1)"
                )
                validation['recommendations'].append(
                    "Possible overfitting or underestimated noise"
                )

        # 3. Parameter correlation validation
        corr_matrix = self.correlation_matrix
        if corr_matrix is not None:
            # Find highly correlated parameters
            n_params = corr_matrix.shape[0]
            high_corr_threshold = 0.95 if strict else 0.99

            for i in range(n_params):
                for j in range(i + 1, n_params):
                    if abs(corr_matrix[i, j]) > high_corr_threshold:
                        validation['warnings'].append(
                            f"Parameters {i} and {j} are highly correlated ({corr_matrix[i, j]:.3f})"
                        )
                        validation['recommendations'].append(
                            "High parameter correlation may indicate overparameterization"
                        )

        # 4. Physical parameter validation
        # Check if parameters are within reasonable ranges
        if self.resonator_params.f_0 is not None:
            freq_range = freq[-1] - freq[0]
            freq_center = (freq[-1] + freq[0]) / 2

            # Resonance should be near the measurement range
            f_0_deviation = abs(self.resonator_params.f_0 - freq_center) / freq_range
            if f_0_deviation > 0.6:  # Resonance more than 60% away from center
                validation['warnings'].append(
                    f"Resonance frequency far from measurement center ({f_0_deviation*100:.1f}% of span)"
                )
                validation['recommendations'].append(
                    "Consider adjusting frequency range to center on resonance"
                )

        if self.resonator_params.kappa is not None and freq.size > 10:
            freq_span = freq[-1] - freq[0]
            # Linewidth should be reasonable compared to frequency span
            if self.resonator_params.kappa > freq_span:
                validation['warnings'].append(
                    "Linewidth larger than frequency span - may be over-broadened"
                )
            elif self.resonator_params.kappa < freq_span / 1000:
                validation['warnings'].append(
                    "Linewidth much smaller than frequency span - may need higher resolution"
                )

        # Set overall status
        if len(validation['warnings']) == 0:
            validation['status'] = 'excellent'
        elif len(validation['warnings']) <= 2:
            validation['status'] = 'good'
        elif len(validation['warnings']) <= 4:
            validation['status'] = 'acceptable'
        else:
            validation['status'] = 'poor'

        return validation

    def to_dict(self):
        """Export fit results to dictionary for saving/serialization"""
        result_dict = {
            'fitted_params': self.resonator_params.params,
            'geometry': None,  # We need to store geometry info
            'param_errors': self.param_errors.tolist() if self.param_errors is not None else None,
            'covariance_matrix': self.pcov.tolist() if self.pcov is not None else None,
            'correlation_matrix': self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
        }
        
        # Try to determine geometry from resonator function
        for geom, func in resonator_dict.items():
            if self.resonator_params.resonator_func == func:
                result_dict['geometry'] = geom
                break
                
        return result_dict
    
    def save_to_file(self, filename):
        """Save fit results to JSON file"""
        import json
        
        result_dict = self.to_dict()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.complexfloating):
                return {'real': float(obj.real), 'imag': float(obj.imag)}
            return obj
        
        # Apply conversion recursively
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        result_dict = deep_convert(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
            
    @classmethod
    def load_from_file(cls, filename):
        """Load fit results from JSON file"""
        import json
        
        with open(filename, 'r') as f:
            result_dict = json.load(f)
        
        # Reconstruct complex numbers
        def reconstruct_complex(obj):
            if isinstance(obj, dict):
                if 'real' in obj and 'imag' in obj:
                    return complex(obj['real'], obj['imag'])
                else:
                    return {k: reconstruct_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [reconstruct_complex(v) for v in obj]
            return obj
        
        result_dict = reconstruct_complex(result_dict)
        
        # Create FitResult object
        params = result_dict['fitted_params']
        geometry = result_dict['geometry']
        pcov = np.array(result_dict['covariance_matrix']) if result_dict['covariance_matrix'] else None
        
        fit_result = cls(params, geometry, pcov=pcov)
        return fit_result

    def compare_with(self, other_fit_result, freq, signal):
        """
        Compare this fit result with another fit result
        
        Args:
            other_fit_result: Another FitResult object
            freq: Frequency array for comparison
            signal: Original signal for comparison
            
        Returns:
            dict: Comparison results
        """
        comparison = {
            'parameter_differences': {},
            'quality_comparison': {},
            'recommendation': None
        }
        
        # Parameter comparison
        param_names = ['f_0', 'kappa', 'kappa_c', 'edelay', 'phi_0']
        
        for param_name in param_names:
            val1 = getattr(self.resonator_params, param_name, None)
            val2 = getattr(other_fit_result.resonator_params, param_name, None)
            
            if val1 is not None and val2 is not None:
                diff = abs(val1 - val2)
                rel_diff = diff / abs(val1) if val1 != 0 else float('inf')
                
                # Get uncertainties if available
                err1 = self.get_param_error(param_name)
                err2 = other_fit_result.get_param_error(param_name)
                
                param_comparison = {
                    'absolute_difference': diff,
                    'relative_difference': rel_diff,
                    'value_1': val1,
                    'value_2': val2,
                    'error_1': err1,
                    'error_2': err2
                }
                
                # Statistical significance test
                if err1 is not None and err2 is not None:
                    combined_error = np.sqrt(err1**2 + err2**2)
                    significance = diff / combined_error if combined_error > 0 else float('inf')
                    param_comparison['statistical_significance'] = significance
                    param_comparison['statistically_different'] = significance > 2  # 2-sigma test
                
                comparison['parameter_differences'][param_name] = param_comparison
        
        # Quality comparison
        gof1 = self.goodness_of_fit(freq, signal)
        gof2 = other_fit_result.goodness_of_fit(freq, signal)
        
        if gof1 is not None and gof2 is not None:
            comparison['quality_comparison'] = {
                'r_squared_1': gof1['r_squared'],
                'r_squared_2': gof2['r_squared'],
                'r_squared_diff': gof1['r_squared'] - gof2['r_squared'],
                'chi_squared_1': gof1['reduced_chi_squared'],
                'chi_squared_2': gof2['reduced_chi_squared'],
                'chi_squared_ratio': gof1['reduced_chi_squared'] / gof2['reduced_chi_squared']
            }
            
            # Recommendation based on quality
            if gof1['r_squared'] > gof2['r_squared'] + 0.01:  # 1% improvement
                comparison['recommendation'] = 'fit_1_better'
            elif gof2['r_squared'] > gof1['r_squared'] + 0.01:
                comparison['recommendation'] = 'fit_2_better'
            else:
                comparison['recommendation'] = 'equivalent_quality'
        
        return comparison

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
