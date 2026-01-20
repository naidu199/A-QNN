"""
Fourier-Based Parameter Estimation
===================================

This module implements Fourier analysis methods for parameter estimation
in quantum circuits. Quantum circuits with parameterized rotation gates
have cost landscapes that can be expressed as Fourier series, enabling
analytical optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from scipy.optimize import minimize_scalar
from scipy.fft import fft, ifft
import warnings


class FourierParameterEstimator:
    """
    Estimates optimal parameters using Fourier analysis of the cost landscape.

    For a quantum circuit with rotation gates, the cost function C(θ) is a
    trigonometric polynomial. This estimator samples the cost at multiple
    points and uses Fourier analysis to find the global minimum.

    The key advantage is that we can find the global minimum exactly
    (within sampling noise) without iterative optimization.

    Mathematical Background:
    For a circuit with L rotation gates each appearing with frequency 1,
    the cost function has the form:
        C(θ) = Σ_k a_k cos(kθ) + b_k sin(kθ)
    where k ranges from -L to L.

    By sampling at 2L+1 uniformly spaced points, we can reconstruct
    the Fourier coefficients exactly and find the minimum analytically.
    """

    def __init__(
        self,
        shots: int = 1024,
        max_frequency: int = 1
    ):
        """
        Initialize the Fourier parameter estimator.

        Args:
            shots: Number of measurement shots per evaluation
            max_frequency: Maximum frequency in the Fourier expansion
                          (usually 1 for single-qubit rotation gates)
        """
        self.shots = shots
        self.max_frequency = max_frequency
        self.total_measurements = 0

    def estimate_fourier_coefficients(
        self,
        circuit: QuantumCircuit,
        parameter: Parameter,
        cost_function: Callable[[QuantumCircuit, Dict], float],
        other_params: Dict[Parameter, float],
        n_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Estimate Fourier coefficients of the cost function.

        Args:
            circuit: Parameterized quantum circuit
            parameter: Parameter whose Fourier coefficients to estimate
            cost_function: Cost function to analyze
            other_params: Fixed values for other parameters
            n_samples: Number of sample points (default: 2*max_frequency + 1)

        Returns:
            Dictionary with Fourier coefficients and analysis results
        """
        if n_samples is None:
            n_samples = 2 * self.max_frequency + 1

        # Sample points uniformly in [0, 2π)
        theta_samples = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        cost_samples = []

        for theta in theta_samples:
            params = {**other_params, parameter: theta}
            cost = cost_function(circuit, params)
            cost_samples.append(cost)
            self.total_measurements += self.shots

        cost_samples = np.array(cost_samples)

        # Compute FFT
        fft_coeffs = fft(cost_samples) / n_samples

        # Extract cosine and sine coefficients
        # a_k = 2 * Re(fft_coeffs[k]) for k > 0
        # b_k = -2 * Im(fft_coeffs[k]) for k > 0
        # a_0 = Re(fft_coeffs[0])

        a_coeffs = np.zeros(self.max_frequency + 1)
        b_coeffs = np.zeros(self.max_frequency + 1)

        a_coeffs[0] = np.real(fft_coeffs[0])

        for k in range(1, self.max_frequency + 1):
            if k < len(fft_coeffs):
                a_coeffs[k] = 2 * np.real(fft_coeffs[k])
                b_coeffs[k] = -2 * np.imag(fft_coeffs[k])

        return {
            'a_coeffs': a_coeffs,
            'b_coeffs': b_coeffs,
            'fft_coeffs': fft_coeffs,
            'theta_samples': theta_samples,
            'cost_samples': cost_samples
        }

    def find_optimal_from_fourier(
        self,
        a_coeffs: np.ndarray,
        b_coeffs: np.ndarray,
        n_candidates: int = 100
    ) -> Tuple[float, float]:
        """
        Find the optimal parameter value from Fourier coefficients.

        Args:
            a_coeffs: Cosine coefficients
            b_coeffs: Sine coefficients
            n_candidates: Number of candidate points to evaluate

        Returns:
            Tuple of (optimal_theta, minimum_cost)
        """
        # Define the reconstructed cost function
        def reconstructed_cost(theta):
            cost = a_coeffs[0]
            for k in range(1, len(a_coeffs)):
                cost += a_coeffs[k] * np.cos(k * theta)
                cost += b_coeffs[k] * np.sin(k * theta)
            return cost

        # For simple case (max_frequency = 1), analytic solution
        if len(a_coeffs) == 2:
            a1, b1 = a_coeffs[1], b_coeffs[1]
            if abs(a1) < 1e-10 and abs(b1) < 1e-10:
                optimal_theta = 0
            else:
                # Critical points where derivative = 0
                # d/dθ (a1 cos(θ) + b1 sin(θ)) = -a1 sin(θ) + b1 cos(θ) = 0
                # tan(θ) = b1/a1
                theta1 = np.arctan2(b1, a1)
                theta2 = theta1 + np.pi

                cost1 = reconstructed_cost(theta1)
                cost2 = reconstructed_cost(theta2)

                optimal_theta = theta1 if cost1 < cost2 else theta2

            return optimal_theta % (2*np.pi), reconstructed_cost(optimal_theta)

        # For higher frequencies, use numerical search
        candidates = np.linspace(0, 2*np.pi, n_candidates)
        costs = [reconstructed_cost(theta) for theta in candidates]

        best_idx = np.argmin(costs)

        # Refine with local optimization
        result = minimize_scalar(
            reconstructed_cost,
            bounds=(candidates[max(0, best_idx-1)],
                   candidates[min(n_candidates-1, best_idx+1)]),
            method='bounded'
        )

        return result.x % (2*np.pi), result.fun

    def estimate_optimal_parameter(
        self,
        circuit: QuantumCircuit,
        parameter: Parameter,
        cost_function: Callable[[QuantumCircuit, Dict], float],
        other_params: Dict[Parameter, float]
    ) -> Tuple[float, float]:
        """
        Estimate the optimal parameter value using Fourier analysis.

        Args:
            circuit: Parameterized quantum circuit
            parameter: Parameter to optimize
            cost_function: Cost function to minimize
            other_params: Fixed values for other parameters

        Returns:
            Tuple of (optimal_theta, minimum_cost)
        """
        fourier_result = self.estimate_fourier_coefficients(
            circuit, parameter, cost_function, other_params
        )

        return self.find_optimal_from_fourier(
            fourier_result['a_coeffs'],
            fourier_result['b_coeffs']
        )

    def analyze_landscape(
        self,
        circuit: QuantumCircuit,
        parameter: Parameter,
        cost_function: Callable[[QuantumCircuit, Dict], float],
        other_params: Dict[Parameter, float],
        n_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Perform comprehensive landscape analysis for a parameter.

        This provides detailed information about the cost landscape,
        including curvature, number of local minima, and sensitivity.

        Args:
            circuit: Parameterized quantum circuit
            parameter: Parameter to analyze
            cost_function: Cost function
            other_params: Fixed parameter values
            n_samples: Number of samples for analysis

        Returns:
            Dictionary with landscape analysis results
        """
        # Get Fourier coefficients with extra samples
        fourier_result = self.estimate_fourier_coefficients(
            circuit, parameter, cost_function, other_params,
            n_samples=n_samples
        )

        a_coeffs = fourier_result['a_coeffs']
        b_coeffs = fourier_result['b_coeffs']
        cost_samples = fourier_result['cost_samples']
        theta_samples = fourier_result['theta_samples']

        # Find optimal
        optimal_theta, optimal_cost = self.find_optimal_from_fourier(
            a_coeffs, b_coeffs
        )

        # Compute landscape properties
        amplitude = np.sqrt(a_coeffs[1]**2 + b_coeffs[1]**2) if len(a_coeffs) > 1 else 0

        # Estimate curvature at optimal point
        # Second derivative of a cos(θ) + b sin(θ) is -a cos(θ) - b sin(θ)
        if len(a_coeffs) > 1:
            curvature = -(a_coeffs[1] * np.cos(optimal_theta) +
                         b_coeffs[1] * np.sin(optimal_theta))
        else:
            curvature = 0

        # Count local minima (sign changes in derivative)
        def derivative(theta):
            d = 0
            for k in range(1, len(a_coeffs)):
                d += -k * a_coeffs[k] * np.sin(k * theta)
                d += k * b_coeffs[k] * np.cos(k * theta)
            return d

        derivatives = [derivative(theta) for theta in theta_samples]
        sign_changes = np.sum(np.diff(np.sign(derivatives)) != 0)
        n_critical_points = sign_changes // 2

        # Variance and flatness
        cost_variance = np.var(cost_samples)
        cost_range = np.max(cost_samples) - np.min(cost_samples)

        # Detect potential barren plateau
        is_flat = cost_range < 0.01 or amplitude < 0.01

        return {
            'optimal_theta': optimal_theta,
            'optimal_cost': optimal_cost,
            'amplitude': amplitude,
            'curvature': curvature,
            'n_critical_points': n_critical_points,
            'cost_variance': cost_variance,
            'cost_range': cost_range,
            'is_flat': is_flat,
            'a_coeffs': a_coeffs,
            'b_coeffs': b_coeffs,
            'mean_cost': np.mean(cost_samples)
        }

    def detect_barren_plateau(
        self,
        circuit: QuantumCircuit,
        parameters: List[Parameter],
        cost_function: Callable[[QuantumCircuit, Dict], float],
        n_random_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Detect if the circuit exhibits barren plateau behavior.

        Barren plateaus are characterized by:
        1. Exponentially vanishing gradients
        2. Cost function concentrated around mean
        3. Very small variance in cost values

        Args:
            circuit: Parameterized quantum circuit
            parameters: Parameters to analyze
            cost_function: Cost function
            n_random_samples: Number of random parameter samples

        Returns:
            Dictionary with barren plateau analysis
        """
        costs = []
        gradients = []

        for _ in range(n_random_samples):
            # Random parameter values
            param_values = {p: np.random.uniform(0, 2*np.pi) for p in parameters}

            cost = cost_function(circuit, param_values)
            costs.append(cost)

            # Estimate gradient magnitude for first parameter
            if parameters:
                param = parameters[0]
                shift = np.pi / 2

                params_plus = {**param_values, param: param_values[param] + shift}
                params_minus = {**param_values, param: param_values[param] - shift}

                cost_plus = cost_function(circuit, params_plus)
                cost_minus = cost_function(circuit, params_minus)

                grad = (cost_plus - cost_minus) / 2
                gradients.append(abs(grad))

        costs = np.array(costs)
        gradients = np.array(gradients) if gradients else np.array([0])

        # Barren plateau indicators
        cost_variance = np.var(costs)
        mean_gradient = np.mean(gradients)
        max_gradient = np.max(gradients)

        # Thresholds for barren plateau detection
        variance_threshold = 0.01
        gradient_threshold = 0.01

        is_barren = cost_variance < variance_threshold and mean_gradient < gradient_threshold

        return {
            'is_barren_plateau': is_barren,
            'cost_variance': cost_variance,
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'mean_cost': np.mean(costs),
            'cost_std': np.std(costs),
            'n_samples': n_random_samples,
            'n_parameters': len(parameters)
        }

    def get_measurement_count(self) -> int:
        """Get total measurements used."""
        return self.total_measurements

    def reset_measurements(self) -> None:
        """Reset measurement counter."""
        self.total_measurements = 0
