"""Resonator parameter classes and quality assessment tools.

This module provides the ResonatorParams class for storing resonator parameters
and the FitResult class for comprehensive fit quality assessment, including
parameter uncertainties, goodness of fit metrics, and validation functionality.
"""

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from .plot import plot
from .utils import get_prefix_str


class FitResult:
    """Container for resonator fit results with uncertainty and quality analysis.

    This class wraps ResonatorParams and adds covariance matrix storage,
    parameter uncertainty calculation, goodness of fit metrics, fit validation,
    and result serialization capabilities.

    Parameters
    ----------
    params : list
        Resonator parameter values.
    geometry : str
        Resonator measurement geometry.
    freq : np.ndarray, optional
        Frequency array used in the fit.
    signal : np.ndarray, optional
        Signal array that was fitted.
    pcov : np.ndarray, optional
        Parameter covariance matrix from fitting.
    fit_func : callable, optional
        Fitted function for quality assessment.

    Attributes
    ----------
    resonator_params : ResonatorParams
        Container for the resonator parameters.
    freq : np.ndarray or None
        Frequency array used in the fit.
    signal : np.ndarray or None
        Signal array that was fitted.
    pcov : np.ndarray or None
        Parameter covariance matrix.
    fit_func : callable or None
        Fitted function.
    """

    def __init__(
        self,
        params: List[float],
        geometry: str,
        freq: Optional[np.ndarray] = None,
        signal: Optional[np.ndarray] = None,
        pcov: Optional[np.ndarray] = None,
        fit_func: Optional[callable] = None,
        original_signal: Optional[np.ndarray] = None,
    ):
        """
        Initialize FitResult with parameters and optional covariance matrix.

        Parameters
        ----------
        params : List[float]
            Resonator parameter values.
        geometry : str
            Resonator measurement geometry.
        freq : np.ndarray, optional
            Frequency array used in the fit.
        signal : np.ndarray, optional
            Signal array that was fitted (after background removal if applied).
        pcov : np.ndarray, optional
            Parameter covariance matrix from fitting.
        fit_func : callable, optional
            Fitted function for quality assessment.
        original_signal : np.ndarray, optional
            Original signal array before any background removal.
        """
        self.resonator_params = ResonatorParams(params, geometry)
        self.freq = freq
        self.signal = signal  # Background-corrected signal used for fitting
        self.original_signal = (
            original_signal  # Original signal before background removal
        )
        self.pcov = pcov
        self.fit_func = fit_func

    @property
    def params(self) -> "ResonatorParams":
        """Access to the ResonatorParams object.

        Returns
        -------
        ResonatorParams
            The resonator parameters container.
        """
        return self.resonator_params

    @property
    def param_errors(self) -> Optional[np.ndarray]:
        """Calculate parameter uncertainties from covariance matrix.

        Returns
        -------
        np.ndarray or None
            Parameter standard errors (square root of diagonal elements
            of covariance matrix), or None if covariance matrix unavailable.
        """
        if self.pcov is not None:
            return np.sqrt(np.diag(self.pcov))
        return None

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Calculate correlation matrix from covariance matrix.

        Returns
        -------
        np.ndarray or None
            Parameter correlation matrix, or None if covariance matrix unavailable.
        """
        if self.pcov is not None:
            std_devs = np.sqrt(np.diag(self.pcov))
            return self.pcov / np.outer(std_devs, std_devs)
        return None

    def get_param_error(self, param_name: str) -> Optional[float]:
        """Get uncertainty for a specific parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter ('f_0', 'kappa', 'kappa_c', 'phi_0',
            're_a_in', 'im_a_in', 'edelay').

        Returns
        -------
        float or None
            Parameter uncertainty, or None if unavailable.
        """
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
        return None

    def goodness_of_fit(self) -> Optional[Dict[str, float]]:
        """Calculate goodness of fit metrics using stored data.

        Computes various statistical measures to assess the quality of the fit,
        including R-squared, reduced chi-squared, and RMS residual.

        Returns
        -------
        dict or None
            Dictionary containing fit quality metrics:
            - 'r_squared': Coefficient of determination (0-1, higher is better)
            - 'reduced_chi_squared': Reduced chi-squared statistic
            - 'rms_residual': Root mean square of residuals
            Returns None if fit function or data is not available.
        """
        if self.fit_func is None or self.freq is None or self.signal is None:
            return None

        fitted_signal = self.fit_func(self.freq, *self.resonator_params.params)
        residuals = self.signal - fitted_signal

        # R-squared
        ss_res = np.sum(np.abs(residuals) ** 2)
        ss_tot = np.sum(np.abs(self.signal - np.mean(self.signal)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Reduced chi-squared
        n_params = len(self.resonator_params.params)
        n_data = len(self.signal)
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

    def validate_fit(self, strict: bool = False) -> Dict[str, Any]:
        """Comprehensive fit validation with automatic warnings and recommendations.

        Performs extensive validation of the fit quality including parameter
        bounds checking, uncertainty analysis, and goodness of fit assessment
        using the stored frequency and signal data.

        Parameters
        ----------
        strict : bool, optional
            If True, apply stricter validation criteria. Default is False.

        Returns
        -------
        dict
            Validation results containing:
            - 'status': Overall validation status ('excellent', 'good', 'warning', 'poor')
            - 'warnings': List of validation warnings
            - 'recommendations': List of improvement recommendations
            - 'metrics': Dictionary of computed validation metrics
        """
        validation = {
            "status": "good",
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        # Check if we have data for validation
        if self.freq is None or self.signal is None:
            validation["warnings"].append(
                "No frequency/signal data available for validation"
            )
            validation["recommendations"].append(
                "Store freq and signal data during fit for comprehensive validation"
            )
            validation["status"] = "limited"
            return validation

        # 1. Parameter uncertainty validation
        if self.param_errors is not None:
            # Check if any parameter has very large uncertainty
            rel_errors = []
            param_names = ["f_0", "kappa_c", "edelay"]

            for param_name in param_names:
                param_val = getattr(self.resonator_params, param_name, None)
                param_err = self.get_param_error(param_name)

                if param_val is not None and param_err is not None and param_val != 0:
                    rel_error = abs(param_err / param_val)
                    rel_errors.append((param_name, rel_error))
                    validation["metrics"][f"{param_name}_relative_error"] = rel_error

                    # Flag parameters with high uncertainty
                    threshold = 0.1 if strict else 0.2  # 10% or 20% relative error
                    if rel_error > threshold:
                        validation["warnings"].append(
                            f"{param_name} has high uncertainty ({rel_error * 100:.1f}%)"
                        )
                        validation["recommendations"].append(
                            f"Consider more data points or lower noise for better {param_name} determination"
                        )  # 2. Goodness of fit validation
        gof = self.goodness_of_fit()
        if gof is not None:
            validation["metrics"].update(gof)

            # R-squared validation
            r2_threshold = 0.95 if strict else 0.9
            if gof["r_squared"] < r2_threshold:
                validation["warnings"].append(
                    f"Low R² = {gof['r_squared']:.3f} (< {r2_threshold})"
                )
                validation["recommendations"].append(
                    "Consider different geometry or check for systematic errors"
                )

            # Reduced chi-squared validation
            if gof["reduced_chi_squared"] > 5:
                validation["warnings"].append(
                    f"High χ²_red = {gof['reduced_chi_squared']:.2f} (>> 1)"
                )
                validation["recommendations"].append(
                    "Systematic residuals detected - model may be inadequate"
                )
            elif gof["reduced_chi_squared"] < 0.1:
                validation["warnings"].append(
                    f"Very low χ²_red = {gof['reduced_chi_squared']:.2f} (<< 1)"
                )
                validation["recommendations"].append(
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
                        validation["warnings"].append(
                            f"Parameters {i} and {j} are highly correlated ({corr_matrix[i, j]:.3f})"
                        )
                        validation["recommendations"].append(
                            "High parameter correlation may indicate overparameterization"
                        )  # 4. Physical parameter validation
        # Check if parameters are within reasonable ranges
        if self.resonator_params.f_0 is not None:
            freq_range = self.freq[-1] - self.freq[0]
            freq_center = (self.freq[-1] + self.freq[0]) / 2

            # Resonance should be near the measurement range
            f_0_deviation = abs(self.resonator_params.f_0 - freq_center) / freq_range
            if f_0_deviation > 0.6:  # Resonance more than 60% away from center
                validation["warnings"].append(
                    f"Resonance frequency far from measurement center ({f_0_deviation * 100:.1f}% of span)"
                )
                validation["recommendations"].append(
                    "Consider adjusting frequency range to center on resonance"
                )

        if self.resonator_params.kappa is not None and self.freq.size > 10:
            freq_span = self.freq[-1] - self.freq[0]
            # Linewidth should be reasonable compared to frequency span
            if self.resonator_params.kappa > freq_span:
                validation["warnings"].append(
                    "Linewidth larger than frequency span - may be over-broadened"
                )
            elif self.resonator_params.kappa < freq_span / 1000:
                validation["warnings"].append(
                    "Linewidth much smaller than frequency span - may need higher resolution"
                )

        # Set overall status
        if len(validation["warnings"]) == 0:
            validation["status"] = "excellent"
        elif len(validation["warnings"]) <= 2:
            validation["status"] = "good"
        elif len(validation["warnings"]) <= 4:
            validation["status"] = "acceptable"
        else:
            validation["status"] = "poor"

        return validation

    def to_dict(self):
        """Export fit results to dictionary for saving/serialization.

        Returns a comprehensive dictionary with named parameters, their values,
        uncertainties, and all fit quality information.
        """
        # Get geometry string
        geometry = None
        for geom, func in resonator_dict.items():
            if self.resonator_params.resonator_func == func:
                geometry = geom
                break
        # Create structured parameter dictionary with names, values, and errors
        parameters = {}
        param_names = [
            "f_0",
            "kappa",
            "kappa_i",
            "kappa_c",
            "edelay",
            "phi_0",
            "Q",
            "Q_i",
            "Q_c",
        ]

        for param_name in param_names:
            param_value = getattr(self.resonator_params, param_name, None)
            if param_value is not None:
                param_error = self.get_param_error(param_name)

                # Handle complex numbers for JSON serialization
                if isinstance(param_value, complex):
                    serializable_value = {
                        "real": float(param_value.real),
                        "imag": float(param_value.imag),
                        "_type": "complex",
                    }
                elif isinstance(param_value, np.ndarray):
                    serializable_value = param_value.tolist()
                elif isinstance(param_value, (np.integer, np.floating)):
                    serializable_value = float(param_value)
                else:
                    serializable_value = param_value

                parameters[param_name] = {
                    "value": serializable_value,
                    "error": float(param_error) if param_error is not None else None,
                    "relative_error": (
                        float(abs(param_error / param_value))
                        if param_error is not None and param_value != 0
                        else None
                    ),
                }

        # Build comprehensive result dictionary
        result_dict = {
            # High-level summary
            "geometry": geometry,
            "fit_summary": self._get_fit_summary(),
            # Structured parameters with names, values, and uncertainties
            "parameters": parameters,
            # Raw arrays for compatibility
            "raw_data": {
                "fitted_params": [
                    (
                        float(x)
                        if not isinstance(x, complex)
                        else {
                            "real": float(x.real),
                            "imag": float(x.imag),
                            "_type": "complex",
                        }
                    )
                    for x in self.resonator_params.params
                ],
                "param_errors": (
                    [float(x) for x in self.param_errors.tolist()]
                    if self.param_errors is not None
                    else None
                ),
                "covariance_matrix": (
                    [[float(x) for x in row] for row in self.pcov.tolist()]
                    if self.pcov is not None
                    else None
                ),
                "correlation_matrix": (
                    [
                        [float(x) for x in row]
                        for row in self.correlation_matrix.tolist()
                    ]
                    if self.correlation_matrix is not None
                    else None
                ),
            },
            # Metadata
            "metadata": {
                "parameter_count": len(self.resonator_params.params),
                "has_uncertainties": self.param_errors is not None,
                "has_covariance": self.pcov is not None,
            },
        }

        return result_dict

    def _get_fit_summary(self):
        """Generate a human-readable summary of the fit results."""
        summary = {
            "geometry": None,
            "resonance_frequency": None,
            "quality_factor": None,
            "linewidth": None,
            "coupling_rates": {},
        }

        # Get geometry name
        for geom, func in resonator_dict.items():
            if self.resonator_params.resonator_func == func:
                summary["geometry"] = geom
                break

        # Add main parameters with proper units and formatting
        if self.resonator_params.f_0 is not None:
            f0_error = self.get_param_error("f_0")
            summary["resonance_frequency"] = {
                "value_hz": self.resonator_params.f_0,
                "error_hz": f0_error,
                "formatted": f"{get_prefix_str(self.resonator_params.f_0, 5)}Hz",
            }

        if self.resonator_params.Q is not None:
            summary["quality_factor"] = {
                "total_q": self.resonator_params.Q,
                "q_internal": self.resonator_params.Q_i,
                "q_coupling": self.resonator_params.Q_c,
            }

        if self.resonator_params.kappa is not None:
            kappa_error = self.get_param_error("kappa")
            summary["linewidth"] = {
                "value_hz": self.resonator_params.kappa,
                "error_hz": kappa_error,
                "formatted": f"{get_prefix_str(self.resonator_params.kappa, 3)}Hz",
            }

        if self.resonator_params.kappa_i is not None:
            summary["coupling_rates"]["internal"] = {
                "value_hz": self.resonator_params.kappa_i,
                "formatted": f"{get_prefix_str(self.resonator_params.kappa_i, 3)}Hz",
            }

        if self.resonator_params.kappa_c is not None:
            summary["coupling_rates"]["external"] = {
                "value_hz": self.resonator_params.kappa_c,
                "formatted": f"{get_prefix_str(self.resonator_params.kappa_c, 3)}Hz",
            }

        if self.resonator_params.phi_0 is not None:
            summary["phase_offset"] = {
                "value_rad": self.resonator_params.phi_0,
                "value_deg": self.resonator_params.phi_0 * 180 / np.pi,
                "error_rad": self.get_param_error("phi_0"),
            }

        if self.resonator_params.edelay is not None:
            summary["electrical_delay"] = {
                "value_s": self.resonator_params.edelay,
                "error_s": self.get_param_error("edelay"),
                "formatted": f"{get_prefix_str(self.resonator_params.edelay, 3)}s",
            }

        return summary

    def save_to_file(self, filename):
        """Save fit results to JSON file."""
        result_dict = self.to_dict()

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.complexfloating):
                return {"real": float(obj.real), "imag": float(obj.imag)}
            return obj

        # Apply conversion recursively
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            return convert_numpy(obj)

        result_dict = deep_convert(result_dict)

        with open(filename, "w") as f:
            json.dump(result_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filename):
        """Load fit results from JSON file."""
        with open(filename) as f:
            result_dict = json.load(f)

        # Reconstruct complex numbers
        def reconstruct_complex(obj):
            if isinstance(obj, dict):
                if "real" in obj and "imag" in obj:
                    return complex(obj["real"], obj["imag"])
                return {k: reconstruct_complex(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [reconstruct_complex(v) for v in obj]
            return obj

        result_dict = reconstruct_complex(result_dict)

        # Create FitResult object - handle both old and new format
        if "fitted_params" in result_dict:
            # Old format
            params = result_dict["fitted_params"]
            geometry = result_dict["geometry"]
            pcov = (
                np.array(result_dict["covariance_matrix"])
                if result_dict["covariance_matrix"]
                else None
            )
        else:
            # New format - extract from raw_data
            raw_data = result_dict["raw_data"]
            params = raw_data["fitted_params"]
            geometry = result_dict["geometry"]
            pcov = (
                np.array(raw_data["covariance_matrix"])
                if raw_data["covariance_matrix"]
                else None
            )

        return cls(params, geometry, pcov=pcov)

    def compare_with(
        self,
        other_fit_result,
        freq: Optional[np.ndarray] = None,
        signal: Optional[np.ndarray] = None,
    ):
        """
        Compare this fit result with another fit result.

        Args:
            other_fit_result: Another FitResult object
            freq: Frequency array for comparison (optional, uses stored data if not provided)
            signal: Original signal for comparison (optional, uses stored data if not provided)

        Returns
        -------
            dict: Comparison results
        """
        comparison = {
            "parameter_differences": {},
            "quality_comparison": {},
            "recommendation": None,
        }

        # Use provided data or fall back to stored data
        freq_to_use = freq if freq is not None else self.freq
        signal_to_use = signal if signal is not None else self.signal

        # Parameter comparison
        param_names = ["f_0", "kappa", "kappa_c", "edelay", "phi_0"]

        for param_name in param_names:
            val1 = getattr(self.resonator_params, param_name, None)
            val2 = getattr(other_fit_result.resonator_params, param_name, None)

            if val1 is not None and val2 is not None:
                diff = abs(val1 - val2)
                rel_diff = diff / abs(val1) if val1 != 0 else float("inf")

                # Get uncertainties if available
                err1 = self.get_param_error(param_name)
                err2 = other_fit_result.get_param_error(param_name)

                param_comparison = {
                    "absolute_difference": diff,
                    "relative_difference": rel_diff,
                    "value_1": val1,
                    "value_2": val2,
                    "error_1": err1,
                    "error_2": err2,
                }

                # Statistical significance test
                if err1 is not None and err2 is not None:
                    combined_error = np.sqrt(err1**2 + err2**2)
                    significance = (
                        diff / combined_error if combined_error > 0 else float("inf")
                    )
                    param_comparison["statistical_significance"] = significance
                    param_comparison["statistically_different"] = (
                        significance > 2
                    )  # 2-sigma test

                comparison["parameter_differences"][param_name] = param_comparison

        # Quality comparison (only if we have data available)
        if freq_to_use is not None and signal_to_use is not None:
            gof1 = self.goodness_of_fit()
            gof2 = other_fit_result.goodness_of_fit()

            if gof1 is not None and gof2 is not None:
                comparison["quality_comparison"] = {
                    "r_squared_1": gof1["r_squared"],
                    "r_squared_2": gof2["r_squared"],
                    "r_squared_diff": gof1["r_squared"] - gof2["r_squared"],
                    "chi_squared_1": gof1["reduced_chi_squared"],
                    "chi_squared_2": gof2["reduced_chi_squared"],
                    "chi_squared_ratio": gof1["reduced_chi_squared"]
                    / gof2["reduced_chi_squared"],
                }

                # Recommendation based on quality
                if gof1["r_squared"] > gof2["r_squared"] + 0.01:  # 1% improvement
                    comparison["recommendation"] = "fit_1_better"
                elif gof2["r_squared"] > gof1["r_squared"] + 0.01:
                    comparison["recommendation"] = "fit_2_better"
                else:
                    comparison["recommendation"] = "equivalent_quality"

        return comparison

    def plot(
        self,
        freq: Optional[np.ndarray] = None,
        signal: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Plot the fit results with original data.

        When background removal was applied, shows both original signal
        (with background) and background-corrected signal for comparison.

        Parameters
        ----------
        freq : np.ndarray, optional
            Frequency array used in the fit. If not provided, uses stored data.
        signal : np.ndarray, optional
            Original signal data that was fitted. If not provided, uses stored data.
        **kwargs
            Additional arguments passed to the plot function.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.

        Examples
        --------
        >>> result = analyze(freq, signal, "transmission")
        >>> fig = result.plot()  # Uses stored data
        >>> fig = result.plot(freq, signal)  # Uses provided data
        >>> fig.show()
        """
        if self.fit_func is None:
            raise ValueError("No fit function available for plotting")

        # Use provided data or fall back to stored data
        freq_to_use = (
            freq if freq is not None else self.freq
        )  # Determine which signal to use for plotting
        # If signal is provided explicitly, use it
        # Otherwise, check if background removal was applied
        if signal is not None:
            signal_to_use = signal
        elif self.original_signal is not None and self.signal is not None:
            # Background removal was applied - show only background-corrected signal
            signal_to_use = self.signal  # Use background-corrected signal only
        else:
            # Use the fitted signal (could be original or background-corrected)
            signal_to_use = self.signal

        if freq_to_use is None or signal_to_use is None:
            error_msg = (
                "No frequency/signal data available for plotting. "
                "Provide freq and signal arguments or ensure data "
                "was stored during fit."
            )
            raise ValueError(error_msg)

        # Calculate fitted signal using the fitted parameters
        fitted_signal = self.fit_func(freq_to_use, *self.resonator_params.params)

        # Standard plotting with background-corrected signal only
        return plot(
            freq_to_use,
            signal_to_use,
            fit=fitted_signal,
            params=self.resonator_params,
            fit_params=self.resonator_params,
            **kwargs,
        )


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


def reflection(freq, f_0, kappa, kappa_c_real, phi_0=0):
    num = 2j * (freq - f_0) + kappa - 2 * kappa_c_real * (1 + 1j * np.tan(phi_0))
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection_mismatched(freq, f_0, kappa, kappa_c_real, phi_0):
    return reflection(freq, f_0, kappa, kappa_c_real, phi_0)


def hanger(freq, f_0, kappa, kappa_c_real, phi_0=0):
    num = 2j * (freq - f_0) + kappa - kappa_c_real * (1 + 1j * np.tan(phi_0))
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


def get_fit_function(geometry, amplitude=True, edelay=True):
    if type(geometry) == str:
        resonator_func = resonator_dict[geometry]
    else:
        resonator_func = geometry

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


class ResonatorParams:
    """Container for resonator parameters with geometry-specific access methods.

    This class provides parameter storage and convenient property-based access
    to resonator parameters based on the measurement geometry. It automatically
    maps parameters to appropriate indices and provides properties for easy
    access to physical quantities like f_0, kappa, Q-factors, etc.

    Parameters
    ----------
    params : list of float
        List of parameter values in geometry-specific order.
    geometry : str
        Resonator measurement geometry identifier:
        - 't' or 'transmission': Transmission measurement
        - 'r' or 'reflection': Reflection measurement
        - 'h' or 'hanger': Hanger measurement
        - 'rm' or 'reflection_mismatched': Reflection with impedance mismatch
        - 'hm' or 'hanger_mismatched': Hanger with impedance mismatch

    Attributes
    ----------
    resonator_func : callable
        Resonator function corresponding to the geometry.
    params : list
        Parameter values.
    f_0_index : int, optional
        Index of resonance frequency parameter.
    kappa_index : int, optional
        Index of total coupling rate parameter.
    kappa_c_real_index : int, optional
        Index of external coupling rate parameter.
    phi_0_index : int, optional
        Index of phase offset parameter.
    re_a_in_index : int, optional
        Index of real part of input amplitude.
    im_a_in_index : int, optional
        Index of imaginary part of input amplitude.
    edelay_index : int, optional
        Index of electrical delay parameter.
    """

    def __init__(self, params: List[float], geometry: str, freq=None, signal=None):
        self.resonator_func = resonator_dict[geometry]
        self.params = params

        self.freq = freq
        self.signal = signal

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
                self.re_a_in_index = 4
                self.im_a_in_index = 5
            if len(self.params) in [5, 7]:
                self.edelay_index = -1

    def tolist(self):
        """Convert parameters to numpy array.

        Returns
        -------
        np.ndarray
            Parameter array.
        """
        return np.array(self.params)

    @property
    def f_0(self):
        """Resonance frequency parameter.

        Returns
        -------
        float or None
            Resonance frequency f_0, or None if not available for this geometry.
        """
        if hasattr(self, "f_0_index"):
            return self.params[self.f_0_index]
        return None

    @property
    def kappa(self):
        """Total coupling rate parameter.

        Returns
        -------
        float or None
            Total coupling rate kappa, or None if not available for this geometry.
        """
        if hasattr(self, "kappa_index"):
            return self.params[self.kappa_index]
        return None

    @property
    def kappa_i(self):
        """Internal coupling rate parameter.

        Returns
        -------
        float or None
            Internal coupling rate kappa_i = kappa - kappa_c, or None if not available.
        """
        if hasattr(self, "kappa_index") and hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_index] - self.params[self.kappa_c_real_index]
        return None

    @property
    def kappa_c_real(self):
        """External coupling rate parameter (real part).

        Returns
        -------
        float or None
            External coupling rate kappa_c, or None if not available for this geometry.
        """
        if hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_c_real_index]
        return None

    @property
    def kappa_c(self):
        """External coupling rate parameter.

        Returns
        -------
        float or None
            External coupling rate kappa_c (alias for kappa_c_real).
        """
        return self.kappa_c_real

    @property
    def Q_i(self):
        """Internal quality factor.

        Returns
        -------
        float or None
            Internal quality factor Q_i = f_0 / kappa_i, or None if not calculable.
        """
        if self.f_0 is not None and self.kappa_i is not None and self.kappa_i != 0:
            return self.f_0 / self.kappa_i
        return None

    @property
    def Q_c(self):
        """External quality factor.

        Returns
        -------
        float or None
            External quality factor Q_c = f_0 / kappa_c, or None if not calculable.
        """
        if self.f_0 is not None and self.kappa_c is not None and self.kappa_c != 0:
            return self.f_0 / self.kappa_c
        return None

    @property
    def Q_total(self):
        """Total quality factor.

        Returns
        -------
        float or None
            Total quality factor Q_total = f_0 / kappa, or None if not calculable.
        """
        if self.f_0 is not None and self.kappa is not None and self.kappa != 0:
            return self.f_0 / self.kappa
        return None

    @property
    def Q(self):
        """Total quality factor (alias for Q_total).

        Returns
        -------
        float or None
            Total quality factor Q = f_0 / kappa.
        """
        return self.Q_total

    @property
    def a_in(self):
        """Complex input amplitude parameter.

        Returns
        -------
        complex or None
            Complex input amplitude a_in = re_a_in + 1j*im_a_in, or None if not available.
        """
        if hasattr(self, "re_a_in_index") and hasattr(self, "im_a_in_index"):
            return (
                self.params[self.re_a_in_index] + 1j * self.params[self.im_a_in_index]
            )
        return None

    @property
    def re_a_in(self):
        """Real part of input amplitude parameter.

        Returns
        -------
        float or None
            Real part of input amplitude, or None if not available.
        """
        a_in = self.a_in
        if a_in is not None:
            return np.real(a_in)
        return None

    @property
    def im_a_in(self):
        """Imaginary part of input amplitude parameter.

        Returns
        -------
        float or None
            Imaginary part of input amplitude, or None if not available.
        """
        a_in = self.a_in
        if a_in is not None:
            return np.imag(a_in)
        return None

    @property
    def edelay(self):
        """Electrical delay parameter.

        Returns
        -------
        float or None
            Electrical delay in seconds, or None if not available for this geometry.
        """
        if hasattr(self, "edelay_index"):
            return self.params[self.edelay_index]
        return None

    @property
    def phi_0(self):
        """Phase offset parameter for mismatched geometries.

        Returns
        -------
        float or None
            Phase offset phi_0 in radians, or None if not available for this geometry.
        """
        if hasattr(self, "phi_0_index"):
            return self.params[self.phi_0_index]
        return None

    def str(
        self,
        latex=False,
        separator=", ",
        precision=2,
        only_f_and_kappa=False,
        f_precision=5,
        red_warning=False,
    ):
        """Generate formatted string representation of resonator parameters.

        Parameters
        ----------
        latex : bool, optional
            If True, use LaTeX formatting for parameter names. Default is False.
        separator : str, optional
            String to use as separator between parameters. Default is ", ".
        precision : int, optional
            Number of significant digits for parameter values. Default is 2.
        only_f_and_kappa : bool, optional
            If True, only include f_0 and kappa parameters. Default is False.
        f_precision : int, optional
            Number of significant digits for frequency. Default is 5.

        Returns
        -------
        str
            Formatted string representation of parameters.
        """
        kappa = {False: "kappa/2pi", True: r"$\kappa/2\pi$"}
        kappa_i = {False: "kappa_i/2pi", True: r"$\kappa_i/2\pi$"}
        kappa_c = {False: "kappa_c/2pi", True: r"$\kappa_c/2\pi$"}
        f_0 = {False: "f_0", True: r"$f_0$"}
        phi_0 = {False: "phi_0", True: r"$\varphi_0$"}

        if self.edelay is not None:
            edelay_str = (
                f"{separator}edelay = {get_prefix_str(self.edelay, precision)}s"
            )
        else:
            edelay_str = ""

        if self.resonator_func == transmission:
            kappa_str = rf"{separator}{kappa[latex]} = {get_prefix_str(self.kappa, precision)}Hz"
        else:
            kappa_str = rf"{separator}{kappa_i[latex]} = {get_prefix_str(self.kappa_i, precision)}Hz{separator}{kappa_c[latex]} = {get_prefix_str(self.kappa_c, precision)}Hz"

        if self.resonator_func in [hanger_mismatched, reflection_mismatched]:
            if red_warning and self.phi_0 is not None and np.abs(self.phi_0) > 0.25:
                phi_0_str = r"%s = %0.2f rad" % (phi_0[latex], self.phi_0)
                phi_0_str = "%s/!\\ " % separator + phi_0_str + " /!\\"
            else:
                phi_0_str = rf"{separator}{phi_0[latex]} = {self.phi_0:0.2f} rad"
        else:
            phi_0_str = ""

        f_0_str = rf"{f_0[latex]} = {get_prefix_str(self.f_0, f_precision)}Hz"

        if only_f_and_kappa:
            return f_0_str + kappa_str
        return f_0_str + kappa_str + phi_0_str + edelay_str

    def __str__(self) -> str:
        """Return string representation of resonator parameters.

        Returns
        -------
        str
            Formatted string showing the resonator parameters.
        """
        return self.str()

    def __repr__(self):
        """Return unambiguous string representation of resonator parameters.

        Returns
        -------
        str
            String representation for debugging and logging.
        """
        return self.str()

    def __call__(self, freq, *args, **kwargs):
        amplitude = self.a_in is not None
        edelay = self.edelay is not None

        fit_func = get_fit_function(self.resonator_func, amplitude, edelay)

        if len(args) == 0 and len(kwargs) == 0:
            params = self.params
        elif len(kwargs) == 0:
            params = np.copy(self.params)
            params[: len(args)] = args
        else:
            resonator = deepcopy(self)
            for key in kwargs:
                resonator.params[resonator.__dict__[key + "_index"]] = kwargs[key]
            resonator.params[: len(args)] = args
            params = resonator.params

        return fit_func(freq, *params)

    def plot(
        self,
        fig=None,
        plot_not_corrected=True,
        font_size=None,
        plot_circle=True,
        center_freq=False,
        only_f_and_kappa=False,
        precision=2,
        alpha_fit=1.0,
        style="Normal",
        title=None,
        params=None,
    ):
        plot(
            self.freq,
            self.signal,
            self(self.freq),
            fig=fig,
            fit_params=self,
            params=params,
            plot_not_corrected=plot_not_corrected,
            font_size=font_size,
            plot_circle=plot_circle,
            center_freq=center_freq,
            only_f_and_kappa=only_f_and_kappa,
            precision=precision,
            alpha_fit=alpha_fit,
            style=style,
            title=title,
        )


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

    f_0 = 3.8e9
    kappa_i = 50e6
    kappa_c = 150e6
    a_in = 1 + 1j
    edelay = 32e-9
    geometry = "r"

    params = [f_0, kappa_i, kappa_c, np.real(a_in), np.imag(a_in), edelay]

    params = ResonatorParams(params, geometry)

    f_0 = 3.8e9
    kappa = 50e6
    a_in = 1 + 1j
    geometry = "t"

    params = [f_0, kappa_i, kappa_c, phi_0, np.real(a_in), np.imag(a_in)]

    params = ResonatorParams(params, geometry)
