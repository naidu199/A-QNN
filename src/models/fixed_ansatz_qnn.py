"""
Fixed Ansatz QNN for Comparison
================================

Implements a traditional fixed-structure Variational Quantum Classifier (VQC)
with gradient-free optimization (COBYLA/SPSA). This serves as the baseline
to compare against the Analytic Iterative Circuit Reconstruction (ARC) method.

The fixed ansatz uses a predetermined circuit structure with:
- Data encoding layer (RY rotations)
- Multiple variational layers (RY + RZ + CNOT entanglement)
- All parameters optimized simultaneously via classical optimizer

This approach is known to suffer from barren plateaus as circuit depth grows.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import time
import warnings


class FixedAnsatzQNN:
    """
    Fixed-structure Variational Quantum Classifier.

    Uses a hardware-efficient ansatz with alternating rotation and
    entanglement layers. Parameters are optimized via COBYLA or other
    gradient-free optimizers.

    This is the standard approach that suffers from barren plateaus,
    used here as a comparison baseline for the ARC method.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        encoding_type: str = 'angle',
        optimizer: str = 'cobyla',
        max_iter: int = 200,
        shots: int = 1024,
        verbose: bool = True,
        subsample_size: int = 200
    ):
        """
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            encoding_type: Data encoding method ('angle', 'iqp')
            optimizer: Classical optimizer ('cobyla', 'nelder-mead', 'powell', 'spsa')
            max_iter: Maximum optimizer iterations
            shots: Number of measurement shots
            verbose: Print progress
            subsample_size: Max training samples per cost evaluation (0=all)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.shots = shots
        self.verbose = verbose
        self.subsample_size = subsample_size

        # Build circuit
        self.data_params = None
        self.var_params = None
        self.circuit = None
        self.optimal_params = None
        self.is_trained = False

        self.cost_history = []
        self.total_measurements = 0

    def _build_circuit(self, n_features: int) -> QuantumCircuit:
        """
        Build the fixed ansatz circuit.

        Structure per layer:
        - RY(θ) rotation on each qubit
        - RZ(θ) rotation on each qubit
        - Linear CNOT entanglement

        Args:
            n_features: Number of input features

        Returns:
            Parameterized quantum circuit
        """
        self.data_params = ParameterVector('x', n_features)

        # Count variational parameters: 2 per qubit per layer (RY + RZ)
        n_var_params = 2 * self.n_qubits * self.n_layers
        self.var_params = ParameterVector('θ', n_var_params)

        qreg = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qreg)

        # Data encoding layer
        for i in range(self.n_qubits):
            circuit.h(qreg[i])
            if i < n_features:
                circuit.ry(self.data_params[i], qreg[i])

        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # RY rotation layer
            for i in range(self.n_qubits):
                circuit.ry(self.var_params[param_idx], qreg[i])
                param_idx += 1

            # RZ rotation layer
            for i in range(self.n_qubits):
                circuit.rz(self.var_params[param_idx], qreg[i])
                param_idx += 1

            # Linear CNOT entanglement
            for i in range(self.n_qubits - 1):
                circuit.cx(qreg[i], qreg[i + 1])

            # Circular connection
            if self.n_qubits > 2:
                circuit.cx(qreg[self.n_qubits - 1], qreg[0])

        self.circuit = circuit
        return circuit

    def _evaluate_expectation(
        self,
        circuit: QuantumCircuit,
        x: np.ndarray,
        var_values: np.ndarray
    ) -> float:
        """
        Evaluate P(|0⟩) for a single data point.

        Args:
            circuit: The quantum circuit
            x: Single data point
            var_values: Variational parameter values

        Returns:
            Probability of measuring |0...0⟩
        """
        param_dict = {}

        # Bind data parameters
        for i in range(min(len(x), len(self.data_params))):
            param_dict[self.data_params[i]] = float(x[i])

        # Bind variational parameters
        for i in range(len(self.var_params)):
            param_dict[self.var_params[i]] = float(var_values[i])

        bound_circuit = circuit.assign_parameters(param_dict)
        sv = Statevector(bound_circuit)
        self.total_measurements += 1
        return sv.probabilities()[0]

    def _cost_function(
        self,
        var_values: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Log-loss cost function for optimization.

        Args:
            var_values: Current variational parameter values
            X: Training features
            y: Training labels

        Returns:
            Log-loss cost
        """
        # Subsample for speed on large datasets
        if self.subsample_size > 0 and len(X) > self.subsample_size:
            idx = np.random.choice(len(X), self.subsample_size, replace=False)
            X_eval, y_eval = X[idx], y[idx]
        else:
            X_eval, y_eval = X, y

        probs = []
        for xi in X_eval:
            p = self._evaluate_expectation(self.circuit, xi, var_values)
            probs.append(p)

        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        cost = log_loss(y_eval, probs)

        self.cost_history.append(cost)
        if self.verbose and len(self.cost_history) % 10 == 0:
            print(f'  Iteration {len(self.cost_history)}: cost = {cost:.6f}', flush=True)

        return cost

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_params: Optional[np.ndarray] = None
    ) -> 'FixedAnsatzQNN':
        """
        Train the fixed ansatz QNN.

        Args:
            X: Training features (N, n_features), preprocessed to [0, π]
            y: Training labels (N,), values 0 or 1
            initial_params: Optional initial parameter values

        Returns:
            self
        """
        n_features = X.shape[1] if X.ndim > 1 else 1

        if self.verbose:
            print(f'Training Fixed Ansatz QNN: {self.n_qubits} qubits, '
                  f'{self.n_layers} layers, optimizer={self.optimizer}')

        # Build circuit
        self._build_circuit(n_features)

        n_var_params = len(self.var_params)
        if self.verbose:
            print(f'  Parameters: {n_var_params}')
            print(f'  Circuit depth: {self.circuit.depth()}')

        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, n_var_params)

        self.cost_history = []

        # Optimize
        t0 = time.time()

        if self.optimizer == 'cobyla':
            result = minimize(
                self._cost_function, initial_params,
                args=(X, y),
                method='COBYLA',
                options={'maxiter': self.max_iter, 'rhobeg': 0.5}
            )
        elif self.optimizer == 'nelder-mead':
            result = minimize(
                self._cost_function, initial_params,
                args=(X, y),
                method='Nelder-Mead',
                options={'maxiter': self.max_iter}
            )
        elif self.optimizer == 'powell':
            result = minimize(
                self._cost_function, initial_params,
                args=(X, y),
                method='Powell',
                options={'maxiter': self.max_iter}
            )
        elif self.optimizer == 'spsa':
            # Simple SPSA implementation
            self.optimal_params = self._spsa_optimize(
                initial_params, X, y, self.max_iter
            )
            self.is_trained = True
            t1 = time.time()
            if self.verbose:
                print(f'  Training time: {t1-t0:.1f}s')
                print(f'  Final cost: {self.cost_history[-1]:.6f}')
            return self
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        self.optimal_params = result.x
        self.is_trained = True

        t1 = time.time()
        if self.verbose:
            print(f'  Training time: {t1-t0:.1f}s')
            print(f'  Final cost: {result.fun:.6f}')
            print(f'  Optimizer iterations: {result.nfev}')

        return self

    def _spsa_optimize(
        self,
        initial_params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        max_iter: int
    ) -> np.ndarray:
        """
        SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.
        """
        params = initial_params.copy()
        a = 0.1
        c = 0.1
        A = max_iter * 0.1
        alpha = 0.602
        gamma = 0.101

        for k in range(max_iter):
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma

            # Random perturbation
            delta = 2 * np.random.binomial(1, 0.5, len(params)) - 1

            # Evaluate at perturbations
            cost_plus = self._cost_function(params + ck * delta, X, y)
            cost_minus = self._cost_function(params - ck * delta, X, y)

            # Gradient estimate
            grad = (cost_plus - cost_minus) / (2 * ck * delta)

            # Update
            params = params - ak * grad

        return params

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using trained circuit.

        Args:
            X: Input features

        Returns:
            (predictions, probabilities)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        probs = []
        for xi in X:
            p = self._evaluate_expectation(self.circuit, xi, self.optimal_params)
            probs.append(p)

        probs = np.array(probs)
        expectations = -1 + 2 * probs
        predictions = np.where(expectations >= 0, 1, 0)

        return predictions, probs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        predictions, _ = self.predict(X)
        return np.mean(predictions == y)

    def get_circuit_info(self) -> Dict[str, Any]:
        """Get circuit information."""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'depth': self.circuit.depth() if self.circuit else 0,
            'n_parameters': len(self.var_params) if self.var_params else 0,
            'encoding_type': self.encoding_type,
            'optimizer': self.optimizer,
            'total_measurements': self.total_measurements
        }

    def get_measurement_count(self) -> int:
        """Get total measurements."""
        return self.total_measurements
