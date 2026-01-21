"""
Adaptive Quantum Neural Network
================================

This module implements the core Adaptive QNN model that addresses the
barren plateau problem through:
1. Incremental circuit construction
2. Analytic parameter estimation
3. Deterministic training without gradient descent
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
import warnings

from ..circuits.circuit_builder import AdaptiveCircuitBuilder
from ..circuits.quantum_gates import QuantumGateSet, create_adaptive_gate_pool
from ..circuits.encoding import DataEncoder
from ..estimators.analytic_estimator import (
    AnalyticParameterEstimator,
    IterativeReconstructionEstimator
)
from ..estimators.fourier_estimator import FourierParameterEstimator


class AdaptiveQNN:
    """
    Adaptive Quantum Neural Network for classification and regression.

    This model implements a novel approach to QNN training that avoids
    barren plateaus by:

    1. **Incremental Construction**: Building the circuit gate-by-gate,
       only adding gates that meaningfully contribute to the objective.

    2. **Analytic Parameter Estimation**: Computing optimal gate parameters
       using closed-form solutions derived from the Fourier structure of
       quantum expectation values.

    3. **Measurement-Efficient Training**: Requiring only O(1) measurements
       per parameter, independent of circuit size.

    The training algorithm works as follows:
    1. Start with a data encoding layer
    2. Evaluate candidate gates from a pool
    3. Add the gate that most improves the objective
    4. Compute its optimal parameter analytically
    5. Repeat until convergence or resource limit

    Attributes:
        n_qubits: Number of qubits
        n_classes: Number of output classes
        circuit_builder: Builds quantum circuits adaptively
        estimator: Computes optimal parameters analytically
        trained_params: Dictionary of trained parameter values
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int = 2,
        encoding_type: str = 'angle',
        max_gates: int = 50,
        shots: int = 1024,
        measurement_budget: int = 50000,
        use_iterative_reconstruction: bool = True,
        backend: Optional[Any] = None
    ):
        """
        Initialize the Adaptive QNN.

        Args:
            n_qubits: Number of qubits in the circuit
            n_classes: Number of output classes
            encoding_type: Data encoding method ('angle', 'amplitude', 'iqp')
            max_gates: Maximum number of gates to add adaptively
            shots: Number of measurement shots per evaluation
            measurement_budget: Total measurement budget for training
            use_iterative_reconstruction: Use iterative reconstruction
            backend: Quantum backend (default: AerSimulator)
        """
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.encoding_type = encoding_type
        self.max_gates = max_gates
        self.shots = shots
        self.measurement_budget = measurement_budget

        # Backend setup
        self.backend = backend or AerSimulator()

        # Initialize components
        self.circuit_builder = AdaptiveCircuitBuilder(
            n_qubits=n_qubits,
            n_classes=n_classes,
            encoding_type=encoding_type
        )

        self.data_encoder = DataEncoder(
            n_qubits=n_qubits,
            encoding_type=encoding_type
        )

        # Choose estimator based on configuration
        if use_iterative_reconstruction:
            self.estimator = IterativeReconstructionEstimator(
                backend=self.backend,
                shots=shots,
                measurement_budget=measurement_budget
            )
        else:
            self.estimator = AnalyticParameterEstimator(
                backend=self.backend,
                shots=shots,
                measurement_budget=measurement_budget
            )

        self.fourier_estimator = FourierParameterEstimator(shots=shots)

        # Gate pool for adaptive construction
        self.gate_pool = create_adaptive_gate_pool(n_qubits)

        # Training state
        self.trained_params: Dict[Parameter, float] = {}
        self.data_params: Optional[ParameterVector] = None
        self.circuit: Optional[QuantumCircuit] = None
        self.is_trained = False
        self.training_history: List[Dict[str, Any]] = []

        # Observables for output
        self._setup_observables()

    def _setup_observables(self) -> None:
        """Setup measurement observables for output extraction."""
        # For classification, measure expectation values of Z operators
        # Note: Qiskit uses little-endian ordering, so qubit 0 is rightmost in Pauli string
        self.observables = []

        for i in range(min(self.n_classes, self.n_qubits)):
            # Z on qubit i - need to reverse position for Qiskit's convention
            z_string = ['I'] * self.n_qubits
            z_string[self.n_qubits - 1 - i] = 'Z'  # Reverse indexing for Qiskit
            pauli_str = ''.join(z_string)
            self.observables.append(SparsePauliOp(pauli_str))

    def _create_cost_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_type: str = 'mse'
    ) -> callable:
        """
        Create a cost function for training.

        Args:
            X: Training features
            y: Training labels
            loss_type: Type of loss ('mse', 'cross_entropy')

        Returns:
            Cost function callable
        """
        def cost_function(circuit: QuantumCircuit, params: Dict) -> float:
            total_loss = 0.0

            for xi, yi in zip(X, y):
                # Get prediction
                pred = self._forward_single(circuit, params, xi)

                # Compute loss
                if loss_type == 'mse':
                    if self.n_classes == 2:
                        target = 1 if yi == 1 else -1
                        loss = (pred[0] - target) ** 2
                    else:
                        # Multi-class: one-hot encoding
                        target = np.zeros(self.n_classes)
                        target[int(yi)] = 1
                        loss = np.sum((pred - target) ** 2)
                else:  # cross_entropy
                    pred_prob = self._expectation_to_probability(pred)
                    loss = -np.log(pred_prob[int(yi)] + 1e-10)

                total_loss += loss

            return total_loss / len(X)

        return cost_function

    def _forward_single(
        self,
        circuit: QuantumCircuit,
        var_params: Dict[Parameter, float],
        x: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass for a single data point.

        Args:
            circuit: Quantum circuit
            var_params: Variational parameter values
            x: Single input data point

        Returns:
            Array of expectation values (one per class)
        """
        # Create parameter binding
        param_binding = {}

        # Get all parameters from the circuit
        circuit_params = circuit.parameters

        # Bind data parameters (those starting with 'x')
        data_params_in_circuit = sorted(
            [p for p in circuit_params if p.name.startswith('x')],
            key=lambda p: p.name
        )

        for i, dp in enumerate(data_params_in_circuit):
            if i < len(x):
                param_binding[dp] = float(x[i])
            else:
                # Bind unused data parameters to 0
                param_binding[dp] = 0.0

        # Bind variational parameters
        param_binding.update(var_params)

        # Bind parameters to circuit
        bound_circuit = circuit.assign_parameters(param_binding)

        # Compute expectation values
        expectations = []

        for obs in self.observables[:self.n_classes]:
            # Use statevector simulation for efficiency
            sv = Statevector(bound_circuit)
            exp_val = np.real(sv.expectation_value(obs))
            expectations.append(exp_val)

        return np.array(expectations)

    def _expectation_to_probability(self, expectations: np.ndarray) -> np.ndarray:
        """Convert expectation values to probabilities."""
        # Map [-1, 1] to [0, 1]
        probs = (expectations + 1) / 2
        # Normalize
        probs = np.clip(probs, 0, 1)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones_like(probs) / len(probs)
        return probs

    def build_initial_circuit(self, n_features: int) -> None:
        """
        Build the initial circuit with data encoding layer.

        Args:
            n_features: Number of input features
        """
        self.circuit_builder.reset()

        # Add encoding layer
        self.data_params = self.circuit_builder.add_encoding_layer()

        self.circuit = self.circuit_builder.get_circuit()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 10,
        improvement_threshold: float = 1e-4,
        verbose: bool = True,
        batch_size: int = 32
    ) -> 'AdaptiveQNN':
        """
        Train the Adaptive QNN using iterative analytic reconstruction.

        This method implements the core adaptive training algorithm:
        1. Build initial circuit with data encoding
        2. Iteratively add gates from the pool
        3. For each gate, compute optimal parameter analytically
        4. Keep gate if it improves the objective

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
            max_iterations: Maximum adaptive construction iterations
            improvement_threshold: Stop if improvement below this
            verbose: Print training progress
            batch_size: Number of samples to use for gate evaluation (faster)

        Returns:
            self (for method chaining)
        """
        n_samples, n_features = X.shape

        if verbose:
            print(f"Training Adaptive QNN with {n_samples} samples, {n_features} features")
            print(f"Configuration: {self.n_qubits} qubits, {self.n_classes} classes")

        # Preprocess data
        X_processed = self.data_encoder.preprocess_data(X, method='minmax')

        # Build initial circuit
        self.build_initial_circuit(n_features)

        # Create cost function for full dataset (used for final evaluation)
        cost_fn_full = self._create_cost_function(X_processed, y)

        # Create mini-batch cost function for faster gate evaluation
        eval_batch_size = min(batch_size, n_samples)
        batch_indices = np.random.choice(n_samples, eval_batch_size, replace=False)
        X_batch = X_processed[batch_indices]
        y_batch = y[batch_indices]
        cost_fn_batch = self._create_cost_function(X_batch, y_batch)

        # Initial cost (on full dataset)
        initial_cost = cost_fn_full(self.circuit, self.trained_params)

        if verbose:
            print(f"Initial cost: {initial_cost:.6f}")

        self.training_history = [{
            'iteration': 0,
            'cost': initial_cost,
            'n_gates': 0,
            'n_params': 0
        }]

        # Adaptive construction loop
        best_cost = initial_cost

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Resample batch each iteration for variety
            batch_indices = np.random.choice(n_samples, eval_batch_size, replace=False)
            X_batch = X_processed[batch_indices]
            y_batch = y[batch_indices]
            cost_fn_batch = self._create_cost_function(X_batch, y_batch)

            # Check measurement budget
            if self.estimator.get_measurement_count() >= self.measurement_budget:
                if verbose:
                    print("Measurement budget exhausted")
                break

            # Find best gate to add
            best_gate = None
            best_param = None
            best_param_value = None
            best_gate_cost = best_cost

            for gate_template in self.gate_pool:
                # Skip if would exceed max gates
                current_n_gates = len(self.circuit_builder.gate_set.gate_history)
                if current_n_gates >= self.max_gates:
                    continue

                # Create trial circuit
                trial_builder = self.circuit_builder.copy()

                # Add candidate gate
                param, _ = trial_builder.add_adaptive_gate(gate_template)
                trial_circuit = trial_builder.get_circuit()

                if param is not None:
                    # Build trial params dict with parameters from the trial circuit
                    # Map existing trained params by name to new circuit's params
                    trial_params = {}
                    trial_circuit_params = {p.name: p for p in trial_circuit.parameters}

                    for orig_param, value in self.trained_params.items():
                        if orig_param.name in trial_circuit_params:
                            trial_params[trial_circuit_params[orig_param.name]] = value

                    landscape_result = self.fourier_estimator.analyze_landscape(
                        trial_circuit, param, cost_fn_batch, trial_params, n_samples=5
                    )
                    optimal_value = landscape_result['optimal_theta']
                    optimal_cost = landscape_result['optimal_cost']

                    if optimal_cost < best_gate_cost:
                        best_gate_cost = optimal_cost
                        best_gate = gate_template
                        best_param = param
                        best_param_value = optimal_value
                else:
                    # Non-parameterized gate
                    # Map params by name for non-parameterized gates too
                    trial_params = {}
                    trial_circuit_params = {p.name: p for p in trial_circuit.parameters}

                    for orig_param, value in self.trained_params.items():
                        if orig_param.name in trial_circuit_params:
                            trial_params[trial_circuit_params[orig_param.name]] = value

                    trial_cost = cost_fn_batch(trial_circuit, trial_params)

                    if trial_cost < best_gate_cost:
                        best_gate_cost = trial_cost
                        best_gate = gate_template
                        best_param = None
                        best_param_value = None

            # Check improvement
            improvement = best_cost - best_gate_cost

            if improvement < improvement_threshold or best_gate is None:
                if verbose:
                    print(f"Converged: improvement {improvement:.6f} < threshold {improvement_threshold}")
                break

            # Add best gate to circuit
            actual_param, _ = self.circuit_builder.add_adaptive_gate(best_gate)
            self.circuit = self.circuit_builder.get_circuit()

            if actual_param is not None and best_param_value is not None:
                self.trained_params[actual_param] = best_param_value

            # Re-evaluate on full dataset for accurate cost tracking
            best_cost = cost_fn_full(self.circuit, self.trained_params)

            # Record history
            self.training_history.append({
                'iteration': iteration + 1,
                'cost': best_cost,
                'n_gates': len(self.circuit_builder.gate_set.gate_history),
                'n_params': len(self.trained_params),
                'gate_added': best_gate['type'],
                'improvement': improvement
            })

            if verbose:
                print(f"Added gate: {best_gate['type']} on qubits {best_gate['qubits']}")
                print(f"Cost: {best_cost:.6f} (improvement: {improvement:.6f})")
                print(f"Circuit depth: {self.circuit.depth()}, Parameters: {len(self.trained_params)}")

        self.is_trained = True

        if verbose:
            print(f"\n=== Training Complete ===")
            print(f"Final cost: {best_cost:.6f}")
            print(f"Total gates: {len(self.circuit_builder.gate_set.gate_history)}")
            print(f"Total parameters: {len(self.trained_params)}")
            print(f"Measurements used: {self.estimator.get_measurement_count()}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X_processed = self.data_encoder.preprocess_data(X, method='minmax')
        predictions = []

        for xi in X_processed:
            exp_vals = self._forward_single(self.circuit, self.trained_params, xi)

            if self.n_classes == 2:
                pred = 1 if exp_vals[0] > 0 else 0
            else:
                pred = np.argmax(exp_vals)

            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X_processed = self.data_encoder.preprocess_data(X, method='minmax')
        probabilities = []

        for xi in X_processed:
            exp_vals = self._forward_single(self.circuit, self.trained_params, xi)
            probs = self._expectation_to_probability(exp_vals)
            probabilities.append(probs)

        return np.array(probabilities)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy on test data.

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_circuit(self) -> QuantumCircuit:
        """Get the trained quantum circuit."""
        return self.circuit

    def get_parameters(self) -> Dict[Parameter, float]:
        """Get trained parameter values."""
        return self.trained_params.copy()

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()

    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the circuit structure."""
        return {
            'n_qubits': self.n_qubits,
            'depth': self.circuit.depth() if self.circuit else 0,
            'n_parameters': len(self.trained_params),
            'n_gates': len(self.circuit_builder.gate_set.gate_history),
            'gate_counts': self.circuit_builder.gate_set.get_gate_count(),
            'encoding_type': self.encoding_type
        }


def create_adaptive_qnn(
    n_qubits: int,
    n_classes: int = 2,
    **kwargs
) -> AdaptiveQNN:
    """
    Factory function to create an Adaptive QNN.

    Args:
        n_qubits: Number of qubits
        n_classes: Number of output classes
        **kwargs: Additional arguments passed to AdaptiveQNN

    Returns:
        Configured AdaptiveQNN instance
    """
    return AdaptiveQNN(
        n_qubits=n_qubits,
        n_classes=n_classes,
        **kwargs
    )
