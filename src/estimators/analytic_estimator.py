"""
Analytic Parameter Estimator for Adaptive QNN
==============================================

This module implements the core innovation of the Adaptive QNN: deterministic
analytic parameter estimation that replaces gradient-based optimization.

The key insight is that for parameterized quantum circuits with certain gate
structures, the optimal parameter values can be computed analytically using
a small number of quantum measurements, avoiding the gradient computation
that leads to barren plateaus.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
import warnings


class AnalyticParameterEstimator:
    """
    Computes optimal gate parameters analytically without gradient descent.

    This estimator leverages the structure of parameterized quantum gates to
    compute optimal parameters directly. For single-qubit rotations of the form
    R(θ) = exp(-i θ G / 2), the expectation value follows a sinusoidal pattern:

        E(θ) = A cos(θ) + B sin(θ) + C

    The optimal θ can be found analytically by measuring at just 3 points.

    This approach:
    1. Eliminates gradient computation → avoids barren plateaus
    2. Requires O(1) measurements per parameter → efficient training
    3. Provides deterministic optimization → stable convergence

    Attributes:
        backend: Quantum backend for circuit execution
        shots: Number of measurement shots per evaluation
        measurement_budget: Total measurement budget
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        shots: int = 1024,
        measurement_budget: int = 10000
    ):
        """
        Initialize the analytic parameter estimator.

        Args:
            backend: Quantum backend (default: AerSimulator)
            shots: Number of shots per circuit evaluation
            measurement_budget: Total measurement budget for training
        """
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.measurement_budget = measurement_budget
        self.total_measurements = 0

    def estimate_optimal_parameter(
        self,
        circuit: QuantumCircuit,
        parameter: Parameter,
        cost_function: Callable[[QuantumCircuit, Dict], float],
        other_params: Dict[Parameter, float]
    ) -> Tuple[float, float]:
        """
        Estimate the optimal value for a single parameter analytically.

        For a rotation gate R(θ), the cost function has the form:
            C(θ) = A cos(θ) + B sin(θ) + C

        We can determine A, B, C by evaluating at θ = 0, π/2, π, then find
        the optimal θ analytically.

        Args:
            circuit: Parameterized quantum circuit
            parameter: The parameter to optimize
            cost_function: Function that evaluates the circuit cost
            other_params: Fixed values for other parameters

        Returns:
            Tuple of (optimal_parameter_value, optimal_cost)
        """
        # Evaluation points for sinusoidal fitting
        eval_points = [0, np.pi/2, np.pi]
        costs = []

        for theta in eval_points:
            params = {**other_params, parameter: theta}
            cost = cost_function(circuit, params)
            costs.append(cost)
            self.total_measurements += self.shots

        # Fit sinusoid: C(θ) = A cos(θ) + B sin(θ) + C
        # At θ=0: C(0) = A + C = costs[0]
        # At θ=π/2: C(π/2) = B + C = costs[1]
        # At θ=π: C(π) = -A + C = costs[2]

        C = (costs[0] + costs[2]) / 2
        A = costs[0] - C
        B = costs[1] - C

        # Optimal θ minimizes A cos(θ) + B sin(θ)
        # Taking derivative: -A sin(θ) + B cos(θ) = 0
        # tan(θ) = B/A → θ = arctan2(B, A)

        if abs(A) < 1e-10 and abs(B) < 1e-10:
            # Flat landscape - parameter doesn't matter
            optimal_theta = 0
        else:
            # Two critical points: θ and θ + π
            theta1 = np.arctan2(B, A)
            theta2 = theta1 + np.pi

            # Evaluate both to find minimum
            cost1 = A * np.cos(theta1) + B * np.sin(theta1) + C
            cost2 = A * np.cos(theta2) + B * np.sin(theta2) + C

            optimal_theta = theta1 if cost1 < cost2 else theta2

        # Compute optimal cost
        optimal_cost = A * np.cos(optimal_theta) + B * np.sin(optimal_theta) + C

        return optimal_theta, optimal_cost

    def estimate_all_parameters(
        self,
        circuit: QuantumCircuit,
        parameters: List[Parameter],
        cost_function: Callable[[QuantumCircuit, Dict], float],
        max_iterations: int = 10,
        tolerance: float = 1e-6
    ) -> Tuple[Dict[Parameter, float], List[float]]:
        """
        Estimate optimal values for all parameters using coordinate descent.

        This method iteratively optimizes each parameter while holding others
        fixed, using the analytic estimation for each parameter.

        Args:
            circuit: Parameterized quantum circuit
            parameters: List of parameters to optimize
            cost_function: Cost function to minimize
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple of (optimal_parameters, cost_history)
        """
        # Initialize parameters randomly
        param_values = {p: np.random.uniform(0, 2*np.pi) for p in parameters}
        cost_history = []

        for iteration in range(max_iterations):
            prev_cost = None

            for param in parameters:
                # Check measurement budget
                if self.total_measurements >= self.measurement_budget:
                    warnings.warn("Measurement budget exceeded")
                    return param_values, cost_history

                # Estimate optimal value for this parameter
                other_params = {p: v for p, v in param_values.items() if p != param}
                optimal_value, cost = self.estimate_optimal_parameter(
                    circuit, param, cost_function, other_params
                )
                param_values[param] = optimal_value

            # Record cost
            current_cost = cost_function(circuit, param_values)
            cost_history.append(current_cost)

            # Check convergence
            if prev_cost is not None and abs(prev_cost - current_cost) < tolerance:
                break
            prev_cost = current_cost

        return param_values, cost_history

    def estimate_with_rotosolve(
        self,
        circuit: QuantumCircuit,
        parameters: List[Parameter],
        cost_function: Callable[[QuantumCircuit, Dict], float],
        initial_values: Optional[Dict[Parameter, float]] = None,
        max_iterations: int = 50
    ) -> Tuple[Dict[Parameter, float], List[float]]:
        """
        Rotosolve algorithm for analytic parameter optimization.

        Rotosolve is a parameter-shift based optimization that finds the
        optimal value of each parameter in closed form, requiring only
        2 function evaluations per parameter per iteration.

        Reference: Ostaszewski et al., "Structure optimization for
        parameterized quantum circuits" (2021)

        Args:
            circuit: Parameterized quantum circuit
            parameters: List of parameters to optimize
            cost_function: Cost function to minimize
            initial_values: Optional initial parameter values
            max_iterations: Maximum iterations

        Returns:
            Tuple of (optimal_parameters, cost_history)
        """
        if initial_values is None:
            param_values = {p: np.random.uniform(0, 2*np.pi) for p in parameters}
        else:
            param_values = initial_values.copy()

        cost_history = []

        for iteration in range(max_iterations):
            for param in parameters:
                # Evaluate at current - π/2 and current + π/2
                current = param_values[param]

                params_minus = {**param_values, param: current - np.pi/2}
                params_plus = {**param_values, param: current + np.pi/2}

                cost_minus = cost_function(circuit, params_minus)
                cost_plus = cost_function(circuit, params_plus)
                self.total_measurements += 2 * self.shots

                # Analytic optimal value
                # C(θ) = A sin(θ + φ) + B
                # Minimum at θ = -φ - π/2

                # From two evaluations at θ ± π/2:
                # cost_plus = A sin(θ + π/2 + φ) + B = A cos(θ + φ) + B
                # cost_minus = A sin(θ - π/2 + φ) + B = -A cos(θ + φ) + B

                A = (cost_plus - cost_minus) / 2

                # Get another point for φ
                cost_current = cost_function(circuit, param_values)
                self.total_measurements += self.shots

                # cost_current = A sin(θ + φ) + B
                # (cost_plus + cost_minus) / 2 = B
                B = (cost_plus + cost_minus) / 2

                # A sin(θ + φ) = cost_current - B
                if abs(A) > 1e-10:
                    sin_val = (cost_current - B) / A
                    sin_val = np.clip(sin_val, -1, 1)
                    theta_plus_phi = np.arcsin(sin_val)
                    phi = theta_plus_phi - current

                    # Optimal θ minimizes, so θ_opt = -φ - π/2
                    optimal_theta = -phi - np.pi/2
                else:
                    optimal_theta = current

                param_values[param] = optimal_theta % (2 * np.pi)

            # Record cost
            current_cost = cost_function(circuit, param_values)
            cost_history.append(current_cost)

            # Early stopping if converged
            if len(cost_history) >= 2:
                if abs(cost_history[-1] - cost_history[-2]) < 1e-7:
                    break

        return param_values, cost_history

    def adaptive_estimation(
        self,
        circuit: QuantumCircuit,
        parameter: Parameter,
        cost_function: Callable[[QuantumCircuit, Dict], float],
        other_params: Dict[Parameter, float],
        n_points: int = 5
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Adaptive estimation with confidence bounds.

        Uses more evaluation points for better estimation in noisy conditions.
        Also provides uncertainty estimates for the optimal parameter.

        Args:
            circuit: Parameterized quantum circuit
            parameter: Parameter to optimize
            cost_function: Cost function
            other_params: Fixed parameter values
            n_points: Number of evaluation points

        Returns:
            Tuple of (optimal_value, optimal_cost, statistics)
        """
        # Uniformly sample points in [0, 2π]
        eval_points = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        costs = []

        for theta in eval_points:
            params = {**other_params, parameter: theta}
            cost = cost_function(circuit, params)
            costs.append(cost)
            self.total_measurements += self.shots

        costs = np.array(costs)

        # Fit sinusoid using least squares
        # C(θ) = A cos(θ) + B sin(θ) + C
        design_matrix = np.column_stack([
            np.cos(eval_points),
            np.sin(eval_points),
            np.ones(n_points)
        ])

        # Least squares fit
        coeffs, residuals, rank, s = np.linalg.lstsq(design_matrix, costs, rcond=None)
        A, B, C = coeffs

        # Optimal θ
        if abs(A) < 1e-10 and abs(B) < 1e-10:
            optimal_theta = 0
            amplitude = 0
        else:
            optimal_theta = np.arctan2(-B, -A)  # Minimum
            amplitude = np.sqrt(A**2 + B**2)

        optimal_cost = A * np.cos(optimal_theta) + B * np.sin(optimal_theta) + C

        # Compute fit quality and uncertainty
        predicted = design_matrix @ coeffs
        mse = np.mean((costs - predicted)**2)
        r_squared = 1 - mse / np.var(costs) if np.var(costs) > 0 else 1.0

        statistics = {
            'amplitude': amplitude,
            'offset': C,
            'mse': mse,
            'r_squared': r_squared,
            'A': A,
            'B': B
        }

        return optimal_theta, optimal_cost, statistics

    def get_measurement_count(self) -> int:
        """Get total number of measurements used."""
        return self.total_measurements

    def reset_measurements(self) -> None:
        """Reset measurement counter."""
        self.total_measurements = 0


class IterativeReconstructionEstimator(AnalyticParameterEstimator):
    """
    Implements iterative analytic reconstruction for circuit parameters.

    This estimator builds the circuit incrementally, adding one gate at a time
    and computing its optimal parameter analytically before moving to the next.
    This approach ensures that:

    1. Each gate is optimally placed given all previous gates
    2. The circuit grows only as needed for the task
    3. Parameters are never updated via gradient descent
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        shots: int = 1024,
        measurement_budget: int = 10000,
        gate_selection_criterion: str = 'cost_reduction'
    ):
        """
        Initialize the iterative reconstruction estimator.

        Args:
            backend: Quantum backend
            shots: Shots per evaluation
            measurement_budget: Total measurement budget
            gate_selection_criterion: How to select next gate
                - 'cost_reduction': Gate that reduces cost most
                - 'gradient_magnitude': Gate with largest gradient
        """
        super().__init__(backend, shots, measurement_budget)
        self.gate_selection_criterion = gate_selection_criterion

    def reconstruct_circuit(
        self,
        base_circuit: QuantumCircuit,
        gate_pool: List[Dict[str, Any]],
        cost_function: Callable[[QuantumCircuit, Dict], float],
        max_gates: int = 20,
        improvement_threshold: float = 1e-4
    ) -> Tuple[QuantumCircuit, Dict[Parameter, float], List[float]]:
        """
        Iteratively reconstruct the circuit by adding optimal gates.

        At each step:
        1. Evaluate all candidate gates from the pool
        2. Select the gate that best improves the cost
        3. Compute its optimal parameter analytically
        4. Add it to the circuit
        5. Repeat until convergence or max gates reached

        Args:
            base_circuit: Starting circuit (e.g., with encoding)
            gate_pool: Pool of candidate gates
            cost_function: Cost function to minimize
            max_gates: Maximum number of gates to add
            improvement_threshold: Stop if improvement is below this

        Returns:
            Tuple of (final_circuit, optimal_parameters, cost_history)
        """
        current_circuit = base_circuit.copy()
        param_values = {}
        cost_history = []

        # Initial cost
        initial_cost = cost_function(current_circuit, param_values)
        cost_history.append(initial_cost)

        for gate_idx in range(max_gates):
            best_gate = None
            best_param = None
            best_cost = float('inf')
            best_param_value = None

            # Evaluate each candidate gate
            for gate_template in gate_pool:
                # Create trial circuit
                trial_circuit = current_circuit.copy()

                # Add gate
                param = self._add_gate_to_circuit(trial_circuit, gate_template)

                if param is not None:
                    # Estimate optimal parameter
                    trial_params = param_values.copy()
                    optimal_value, optimal_cost, _ = self.adaptive_estimation(
                        trial_circuit, param, cost_function, trial_params
                    )

                    if optimal_cost < best_cost:
                        best_cost = optimal_cost
                        best_gate = gate_template
                        best_param = param
                        best_param_value = optimal_value
                else:
                    # Non-parameterized gate
                    trial_cost = cost_function(trial_circuit, param_values)

                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_gate = gate_template
                        best_param = None
                        best_param_value = None

            # Check for improvement
            if best_gate is None or best_cost >= cost_history[-1] - improvement_threshold:
                break

            # Add best gate to circuit
            best_param_actual = self._add_gate_to_circuit(current_circuit, best_gate)

            if best_param_actual is not None and best_param_value is not None:
                param_values[best_param_actual] = best_param_value

            cost_history.append(best_cost)

        return current_circuit, param_values, cost_history

    def _add_gate_to_circuit(
        self,
        circuit: QuantumCircuit,
        gate_template: Dict[str, Any]
    ) -> Optional[Parameter]:
        """
        Add a gate from a template to the circuit.

        Args:
            circuit: Circuit to modify
            gate_template: Gate specification

        Returns:
            Parameter if gate is parameterized, None otherwise
        """
        gate_type = gate_template['type']
        qubits = gate_template['qubits']

        param = None

        if gate_template.get('parameterized', False):
            param = Parameter(f'θ_{len(circuit.parameters)}')

            if gate_type in ['rx', 'ry', 'rz']:
                getattr(circuit, gate_type)(param, qubits[0])
            elif gate_type in ['crx', 'cry', 'crz']:
                getattr(circuit, gate_type)(param, qubits[0], qubits[1])
        else:
            if gate_type == 'cx':
                circuit.cx(qubits[0], qubits[1])
            elif gate_type == 'cz':
                circuit.cz(qubits[0], qubits[1])
            elif gate_type == 'h':
                circuit.h(qubits[0])

        return param
