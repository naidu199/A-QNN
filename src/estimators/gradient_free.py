"""
Gradient-Free Optimization Methods
===================================

This module provides gradient-free optimization methods for quantum neural
networks. These methods are particularly useful for avoiding barren plateaus
and for optimization in noisy quantum environments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from scipy.optimize import minimize, differential_evolution
import warnings


class GradientFreeOptimizer:
    """
    Gradient-free optimization methods for quantum circuits.

    This class provides several optimization algorithms that do not
    require gradient computation, making them suitable for:
    1. Circuits with barren plateaus
    2. Noisy quantum hardware
    3. Discrete or hybrid optimization problems

    Available Methods:
    - COBYLA: Constrained optimization by linear approximation
    - Nelder-Mead: Simplex-based direct search
    - Powell: Direction set method
    - Differential Evolution: Evolutionary algorithm
    - SPSA: Simultaneous perturbation stochastic approximation
    """

    SUPPORTED_METHODS = [
        'cobyla', 'nelder-mead', 'powell',
        'differential_evolution', 'spsa', 'cma-es'
    ]

    def __init__(
        self,
        method: str = 'cobyla',
        shots: int = 1024,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Initialize the gradient-free optimizer.

        Args:
            method: Optimization method to use
            shots: Number of measurement shots per evaluation
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.method = method.lower()
        self.shots = shots
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.total_evaluations = 0
        self.cost_history: List[float] = []

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Supported: {self.SUPPORTED_METHODS}")

    def optimize(
        self,
        circuit: QuantumCircuit,
        parameters: List[Parameter],
        cost_function: Callable[[QuantumCircuit, Dict], float],
        initial_values: Optional[Dict[Parameter, float]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[Dict[Parameter, float], List[float]]:
        """
        Optimize circuit parameters using gradient-free method.

        Args:
            circuit: Parameterized quantum circuit
            parameters: List of parameters to optimize
            cost_function: Cost function to minimize
            initial_values: Optional initial parameter values
            bounds: Optional bounds for each parameter

        Returns:
            Tuple of (optimal_parameters, cost_history)
        """
        self.cost_history.clear()
        self.total_evaluations = 0

        # Initialize
        if initial_values is None:
            x0 = np.random.uniform(0, 2*np.pi, len(parameters))
        else:
            x0 = np.array([initial_values.get(p, np.random.uniform(0, 2*np.pi))
                          for p in parameters])

        if bounds is None:
            bounds = [(0, 2*np.pi)] * len(parameters)

        # Wrapper for cost function
        def objective(x):
            param_dict = {p: v for p, v in zip(parameters, x)}
            cost = cost_function(circuit, param_dict)
            self.cost_history.append(cost)
            self.total_evaluations += 1
            return cost

        # Run optimization
        if self.method == 'cobyla':
            result = self._optimize_cobyla(objective, x0, bounds)
        elif self.method == 'nelder-mead':
            result = self._optimize_nelder_mead(objective, x0)
        elif self.method == 'powell':
            result = self._optimize_powell(objective, x0, bounds)
        elif self.method == 'differential_evolution':
            result = self._optimize_de(objective, bounds)
        elif self.method == 'spsa':
            result = self._optimize_spsa(objective, x0)
        elif self.method == 'cma-es':
            result = self._optimize_cma_es(objective, x0, bounds)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Convert to parameter dictionary
        optimal_params = {p: v for p, v in zip(parameters, result)}

        return optimal_params, self.cost_history

    def _optimize_cobyla(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """COBYLA optimization."""
        # Convert bounds to constraints
        constraints = []
        for i, (lb, ub) in enumerate(bounds):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, lb=lb: x[i] - lb})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, ub=ub: ub - x[i]})

        result = minimize(
            objective,
            x0,
            method='COBYLA',
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'rhobeg': 0.5}
        )

        return result.x

    def _optimize_nelder_mead(
        self,
        objective: Callable,
        x0: np.ndarray
    ) -> np.ndarray:
        """Nelder-Mead simplex optimization."""
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': self.max_iterations,
                'xatol': self.tolerance,
                'fatol': self.tolerance
            }
        )

        return result.x

    def _optimize_powell(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Powell's method optimization."""
        result = minimize(
            objective,
            x0,
            method='Powell',
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        return result.x

    def _optimize_de(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Differential evolution optimization."""
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.max_iterations,
            tol=self.tolerance,
            seed=42,
            polish=True
        )

        return result.x

    def _optimize_spsa(
        self,
        objective: Callable,
        x0: np.ndarray,
        a: float = 0.1,
        c: float = 0.1,
        A: float = 10
    ) -> np.ndarray:
        """
        SPSA (Simultaneous Perturbation Stochastic Approximation).

        SPSA estimates the gradient using only 2 function evaluations
        per iteration, regardless of dimension.
        """
        x = x0.copy()
        n = len(x0)

        for k in range(1, self.max_iterations + 1):
            # Gain sequences
            ak = a / (k + A) ** 0.602
            ck = c / k ** 0.101

            # Random perturbation direction
            delta = np.random.choice([-1, 1], size=n)

            # Gradient approximation
            x_plus = x + ck * delta
            x_minus = x - ck * delta

            cost_plus = objective(x_plus)
            cost_minus = objective(x_minus)

            g_hat = (cost_plus - cost_minus) / (2 * ck * delta)

            # Update
            x = x - ak * g_hat

            # Project to [0, 2π]
            x = np.mod(x, 2 * np.pi)

            # Early stopping
            if len(self.cost_history) >= 10:
                recent = self.cost_history[-10:]
                if np.std(recent) < self.tolerance:
                    break

        return x

    def _optimize_cma_es(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        sigma0: float = 0.5
    ) -> np.ndarray:
        """
        CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

        A sophisticated evolutionary algorithm that adapts the search
        distribution based on successful steps.
        """
        try:
            import cma

            # Scale to handle bounds
            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])

            options = {
                'bounds': [lb.tolist(), ub.tolist()],
                'maxfevals': self.max_iterations * len(x0),
                'tolfun': self.tolerance,
                'verbose': -9  # Suppress output
            }

            result = cma.fmin(
                objective,
                x0,
                sigma0,
                options=options
            )

            return result[0]

        except ImportError:
            warnings.warn("CMA-ES requires 'cma' package. Falling back to differential evolution.")
            return self._optimize_de(objective, bounds)

    def get_evaluation_count(self) -> int:
        """Get total number of function evaluations."""
        return self.total_evaluations


class HybridOptimizer:
    """
    Combines analytic estimation with gradient-free fine-tuning.

    This optimizer uses a two-phase approach:
    1. Phase 1: Use analytic estimation for initial parameter values
    2. Phase 2: Fine-tune with gradient-free optimization

    This combines the benefits of both approaches:
    - Fast convergence from analytic methods
    - Robustness from gradient-free optimization
    """

    def __init__(
        self,
        analytic_estimator: Any,
        gradient_free_optimizer: GradientFreeOptimizer,
        analytic_iterations: int = 5,
        fine_tune_iterations: int = 20
    ):
        """
        Initialize the hybrid optimizer.

        Args:
            analytic_estimator: Analytic parameter estimator
            gradient_free_optimizer: Gradient-free optimizer for fine-tuning
            analytic_iterations: Iterations for analytic phase
            fine_tune_iterations: Iterations for fine-tuning phase
        """
        self.analytic_estimator = analytic_estimator
        self.gradient_free_optimizer = gradient_free_optimizer
        self.analytic_iterations = analytic_iterations
        self.fine_tune_iterations = fine_tune_iterations

    def optimize(
        self,
        circuit: QuantumCircuit,
        parameters: List[Parameter],
        cost_function: Callable[[QuantumCircuit, Dict], float]
    ) -> Tuple[Dict[Parameter, float], List[float]]:
        """
        Two-phase optimization.

        Args:
            circuit: Parameterized quantum circuit
            parameters: Parameters to optimize
            cost_function: Cost function to minimize

        Returns:
            Tuple of (optimal_parameters, cost_history)
        """
        cost_history = []

        # Phase 1: Analytic estimation
        analytic_params, analytic_history = self.analytic_estimator.estimate_all_parameters(
            circuit, parameters, cost_function,
            max_iterations=self.analytic_iterations
        )
        cost_history.extend(analytic_history)

        # Phase 2: Fine-tune with gradient-free
        self.gradient_free_optimizer.max_iterations = self.fine_tune_iterations
        fine_tune_params, fine_tune_history = self.gradient_free_optimizer.optimize(
            circuit, parameters, cost_function,
            initial_values=analytic_params
        )
        cost_history.extend(fine_tune_history)

        return fine_tune_params, cost_history
