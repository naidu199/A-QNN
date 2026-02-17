"""
Analytic Iterative Circuit Reconstruction (ARC) Estimator
==========================================================

Implements the core algorithm from:
"Mitigating Barren Plateaus in Quantum Neural Networks via
Analytic Iterative Circuit Reconstruction"

The key idea: instead of optimizing all parameters simultaneously via
gradient descent (which suffers from barren plateaus), we build the circuit
gate-by-gate. For each candidate gate, we evaluate the cost function at
3 specific angles (0, π/2, −π/2) to analytically determine the sinusoidal
cost landscape C(θ) = a·cos(θ·Δx − b) + c, then pick the optimal θ.

This is the Rotosolve-like approach applied at circuit construction time.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from sklearn.metrics import log_loss
import time
import warnings


def determine_sine_curve(f0: float, fp: float, fm: float) -> Tuple[float, float, float]:
    """
    Determine a, b, c values for a·cos(θ − b) + c
    by solving the system for f0=f(0), fp=f(π/2), fm=f(−π/2).

    Rotosolve-style reconstruction adapted from the reference paper.

    Args:
        f0: function value at θ=0
        fp: function value at θ=+π/2
        fm: function value at θ=−π/2

    Returns:
        (a, b, c): amplitude, shift, offset of the cosine function
    """
    c = 0.5 * (fp + fm)
    b = np.arctan2(2 * (f0 - c), fp - fm)
    a = np.sqrt((f0 - c) ** 2 + 0.25 * (fp - fm) ** 2)
    b = -b + np.pi / 2
    return a, b, c


class ARCGatePool:
    """
    Gate pool for Analytic Reconstruction Circuit building.

    Each gate in the pool is described as a descriptor string with format:
    'GateType_featureID' where featureID determines which data feature
    multiplies the trainable rotation angle.
    """

    # Standard gate types available
    SINGLE_QUBIT = ['U1', 'U2', 'U3', 'H', 'Rz', 'Rx', 'Ry']
    TWO_QUBIT = ['X', 'Z', 'Xn', 'Zn']  # CNOT, CZ, skip-CNOT, skip-CZ
    PARAMETERIZED_ONLY = ['P', 'CP']

    def __init__(self, gate_list: List[str], n_qubits: int):
        """
        Args:
            gate_list: List of gate type strings (e.g., ['U1', 'H', 'X', 'Ry'])
            n_qubits: Number of qubits
        """
        self.gate_list = gate_list
        self.n_qubits = n_qubits
        self.pool = self._build_pool()

    def _build_pool(self) -> List[np.ndarray]:
        """
        Build the matrix of all possible single-gate placements.
        Each entry is a vector of length n_qubits where one position
        has the gate descriptor and all others have '111111' (no-op).
        """
        matrix_feature = []
        for gate_type in self.gate_list:
            for qubit in range(self.n_qubits):
                vector = np.full(self.n_qubits, '111111', dtype=object)
                vector[qubit] = f'{gate_type}_x'
                matrix_feature.append(vector)
        return matrix_feature

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, idx):
        return self.pool[idx]


def build_gate_circuit(
    gate_descriptor: np.ndarray,
    n_qubits: int,
    angle: float,
    params: ParameterVector,
    data_feature: int = 0
) -> QuantumCircuit:
    """
    Build a single-layer quantum circuit from a gate descriptor.

    The gate descriptor specifies which gate to place on which qubit.
    The angle is the trainable weight, and the data feature determines
    which input feature multiplies the rotation.

    Args:
        gate_descriptor: Array of gate strings, one per qubit
        n_qubits: Number of qubits
        angle: Rotation angle (trainable parameter)
        params: Parameter vector for data encoding
        data_feature: Which feature index to use for data-dependent rotation

    Returns:
        Single-layer quantum circuit
    """
    qreg = QuantumRegister(n_qubits, 'q')
    circuit = QuantumCircuit(qreg)

    for j in range(n_qubits):
        desc_parts = str(gate_descriptor[j]).split('_')
        if len(desc_parts) != 2:
            continue

        gate_type, feat_id = desc_parts

        # Determine the parameter: data feature or fixed
        if feat_id == 'x' or feat_id == 'y':
            param_val = params[data_feature] if data_feature < len(params) else 1.0
        else:
            param_val = params[int(feat_id)] if int(feat_id) < len(params) else 1.0

        # Apply gate based on type
        if gate_type == 'U1':
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'U2':
            circuit.rx(angle * param_val, qreg[j])
        elif gate_type == 'U3':
            circuit.ry(angle * param_val, qreg[j])
        elif gate_type == 'H':
            circuit.h(qreg[j])
        elif gate_type == 'Rz':
            circuit.rz(angle, qreg[j])
        elif gate_type == 'Rx':
            circuit.rx(angle, qreg[j])
        elif gate_type == 'Ry':
            circuit.ry(angle, qreg[j])
        elif gate_type == 'P':
            circuit.p(angle * param_val, qreg[j])
        elif gate_type == 'X' and j >= 1:
            circuit.cx(qreg[j-1], qreg[j])
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'Xn' and j >= 2:
            circuit.cx(qreg[j-2], qreg[j])
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'Z' and j >= 1:
            circuit.cz(qreg[j-1], qreg[j])
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'Zn' and j >= 2:
            circuit.cz(qreg[j-2], qreg[j])
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'CP' and j >= 1:
            circuit.cp(angle * param_val, qreg[j-1], qreg[j])

    return circuit


class ARCEstimator:
    """
    Analytic Iterative Circuit Reconstruction Estimator.

    Implements the paper's algorithm:
    1. Start with empty circuit
    2. For each candidate gate from the pool:
       a. Evaluate expectation at θ=0, θ=π/2, θ=−π/2
       b. Fit sinusoidal: f(θ) = a·cos(θ·x_i − b) + c
       c. Find optimal θ analytically by minimizing cost
    3. Select the gate+angle that gives lowest cost
    4. Add it to the circuit
    5. Repeat until convergence

    The cost function is the log-loss for QNN classification:
    C = (1/N) Σ_i LogLoss(y_i, P(x_i))

    where P(x_i) = |⟨0|U(x_i)|0⟩|² is the probability of measuring |0⟩.
    """

    def __init__(
        self,
        n_qubits: int,
        gate_list: List[str] = None,
        num_samples_classical: int = 1000,
        convergence_threshold: float = 1e-8,
        max_gates: int = 150,
        verbose: bool = True
    ):
        """
        Args:
            n_qubits: Number of qubits
            gate_list: Gate types to use (default: ['U1', 'U2', 'U3', 'H', 'X', 'Z'])
            num_samples_classical: Number of θ samples for classical optimization
            convergence_threshold: Stop when cost improvement < this
            max_gates: Maximum number of gates to add
            verbose: Print progress
        """
        self.n_qubits = n_qubits
        self.gate_list = gate_list or ['U1', 'U2', 'U3', 'H', 'X', 'Z']
        self.num_samples_classical = num_samples_classical
        self.convergence_threshold = convergence_threshold
        self.max_gates = max_gates
        self.verbose = verbose

        # Build gate pool
        self.gate_pool = ARCGatePool(self.gate_list, n_qubits)

        # Training state
        self.circuit = None
        self.gate_sequence = []  # List of (gate_descriptor, angle, feature)
        self.cost_history = []
        self.total_measurements = 0

    def _initialize_circuit(self) -> QuantumCircuit:
        """Create an empty circuit."""
        qreg = QuantumRegister(self.n_qubits, 'qc')
        return QuantumCircuit(qreg)

    def _evaluate_expectation(
        self,
        circuit: QuantumCircuit,
        x: np.ndarray,
        params: ParameterVector
    ) -> float:
        """
        Evaluate the probability of measuring |0...0⟩ for a single data point.

        Args:
            circuit: The quantum circuit
            x: Single data point features
            params: Parameter vector for data encoding

        Returns:
            Probability P(|0⟩) = |⟨0|U(x)|0⟩|²
        """
        qc = circuit.copy()

        # Bind data parameters
        param_dict = {}
        circuit_params = {p.name: p for p in qc.parameters}
        for i in range(min(len(x), len(params))):
            if params[i].name in circuit_params:
                param_dict[circuit_params[params[i].name]] = float(x[i])

        # Bind any remaining unbound parameters to 0
        for p in qc.parameters:
            if p not in param_dict:
                param_dict[p] = 0.0

        if param_dict:
            qc = qc.assign_parameters(param_dict)

        sv = Statevector(qc)
        prob_zero = sv.probabilities()[0]
        self.total_measurements += 1
        return prob_zero

    def _compute_cost(
        self,
        circuit: QuantumCircuit,
        X: np.ndarray,
        y: np.ndarray,
        params: ParameterVector
    ) -> float:
        """
        Compute log-loss cost over the training set.

        Args:
            circuit: Quantum circuit
            X: Training features (N, n_features)
            y: Training labels (N,) with values 0 or 1
            params: Parameter vector

        Returns:
            Log-loss cost
        """
        probs = []
        for xi in X:
            p = self._evaluate_expectation(circuit, xi, params)
            probs.append(p)

        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return log_loss(y, probs)

    def _evaluate_gate_candidate(
        self,
        base_circuit: QuantumCircuit,
        gate_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        params: ParameterVector,
        count: int
    ) -> Tuple[List[float], List[float]]:
        """
        Evaluate a candidate gate across all features using the 3-point
        sinusoidal reconstruction.

        For each data point, we evaluate the circuit+gate at θ=0, π/2, −π/2
        to get a, b, c of the sinusoidal cost landscape. Then we search
        for the optimal θ across a classical grid.

        Args:
            base_circuit: Current circuit before adding gate
            gate_idx: Index into gate pool
            X: Training data
            y: Training labels
            params: Parameter vector
            count: Current gate count (0 = first gate)

        Returns:
            (optimal_thetas, optimal_costs) for each feature
        """
        gate_desc = self.gate_pool[gate_idx]
        n_features = len(params)
        n_samples = len(X)

        theta_test = [0, np.pi * 0.5, -np.pi * 0.5]

        opt_thetas = []
        opt_costs = []

        for feat in range(n_features):
            # For each data point, compute a, b, c of the sinusoidal fit
            AA = np.zeros(n_samples)
            BB = np.zeros(n_samples)
            CC = np.zeros(n_samples)

            if count == 0:
                # First gate: no data dependence yet, same a,b,c for all
                expect_test = [0, 0, 0]
                for kt in range(3):
                    # Build test circuit with test angle
                    gate_desc_feat = gate_desc.copy()
                    for q in range(self.n_qubits):
                        if gate_desc_feat[q] != '111111':
                            g, _ = str(gate_desc_feat[q]).split('_')
                            gate_desc_feat[q] = f'{g}_{feat}'

                    angle_list = [theta_test[kt]] * self.n_qubits
                    test_gate = build_gate_circuit(
                        gate_desc_feat, self.n_qubits, theta_test[kt],
                        ParameterVector('t', n_features), feat
                    )

                    # Assign param[feat]=1 (test mode)
                    test_params = {p: 1.0 for p in test_gate.parameters}
                    if test_params:
                        test_gate = test_gate.assign_parameters(test_params)

                    test_circuit = base_circuit.copy()
                    test_circuit.compose(test_gate, qubits=list(range(self.n_qubits)), inplace=True)

                    # Remove remaining unbound params
                    remaining = {p: 0.0 for p in test_circuit.parameters}
                    if remaining:
                        test_circuit = test_circuit.assign_parameters(remaining)

                    sv = Statevector(test_circuit)
                    expect_test[kt] = sv.probabilities()[0]
                    self.total_measurements += 1

                a, b, c = determine_sine_curve(expect_test[0], expect_test[1], expect_test[2])
                AA[:] = a
                BB[:] = b
                CC[:] = c
            else:
                # Subsequent gates: data-dependent a,b,c per data point
                for i in range(n_samples):
                    expect_test = [0, 0, 0]
                    for kt in range(3):
                        gate_desc_feat = gate_desc.copy()
                        for q in range(self.n_qubits):
                            if gate_desc_feat[q] != '111111':
                                g, _ = str(gate_desc_feat[q]).split('_')
                                gate_desc_feat[q] = f'{g}_{feat}'

                        test_gate = build_gate_circuit(
                            gate_desc_feat, self.n_qubits, theta_test[kt],
                            ParameterVector('t', n_features), feat
                        )

                        # Assign data feature value
                        test_params = {}
                        for p in test_gate.parameters:
                            test_params[p] = float(X[i][feat]) if 'x' not in p.name else float(X[i][feat])
                        if not test_params:
                            test_params = {p: 1.0 for p in test_gate.parameters}
                        if test_params:
                            test_gate = test_gate.assign_parameters(test_params)

                        # Build full circuit with data
                        test_circuit = base_circuit.copy()

                        # Bind existing data params
                        existing_params = {}
                        for p in test_circuit.parameters:
                            for fi in range(n_features):
                                if p.name == params[fi].name:
                                    existing_params[p] = float(X[i][fi])
                        remaining = {p: 0.0 for p in test_circuit.parameters if p not in existing_params}
                        existing_params.update(remaining)
                        if existing_params:
                            test_circuit = test_circuit.assign_parameters(existing_params)

                        test_circuit.compose(test_gate, qubits=list(range(self.n_qubits)), inplace=True)

                        # Remove any remaining params
                        final_params = {p: 0.0 for p in test_circuit.parameters}
                        if final_params:
                            test_circuit = test_circuit.assign_parameters(final_params)

                        sv = Statevector(test_circuit)
                        expect_test[kt] = sv.probabilities()[0]
                        self.total_measurements += 1

                    a, b, c = determine_sine_curve(expect_test[0], expect_test[1], expect_test[2])
                    AA[i] = a
                    BB[i] = b
                    CC[i] = c

            # Find optimal theta by grid search on the reconstructed cost
            opt_theta, opt_cost = self._find_optimal_theta(
                AA, BB, CC, X, y, feat
            )
            opt_thetas.append(opt_theta)
            opt_costs.append(opt_cost)

        return opt_thetas, opt_costs

    def _find_optimal_theta(
        self,
        AA: np.ndarray,
        BB: np.ndarray,
        CC: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        feature: int
    ) -> Tuple[float, float]:
        """
        Find the optimal angle θ that minimizes the reconstructed log-loss.

        For each θ, the expected probability for data point i is:
        P_i(θ) = AA[i]·cos(θ·x_i[feature] − BB[i]) + CC[i]

        We minimize: C(θ) = LogLoss(y, P(θ))

        Args:
            AA, BB, CC: Sinusoidal parameters per data point
            X: Training features
            y: Training labels
            feature: Which feature dimension

        Returns:
            (optimal_theta, optimal_cost)
        """
        num_xs = int(self.num_samples_classical)
        if num_xs % 2 == 0:
            num_xs += 1
        xs = np.linspace(-1, 1, num_xs)

        # Vectorized computation
        # expect[th, i] = AA[i] * cos(xs[th] * X[i, feature] - BB[i]) + CC[i]
        x_feat = X[:, feature] if X.ndim > 1 else X

        # Shape: (num_xs, n_samples)
        expect = AA[None, :] * np.cos(
            xs[:, None] * x_feat[None, :] - BB[None, :]
        ) + CC[None, :]

        # Clip for numerical stability
        expect = np.clip(expect, 1e-15, 1 - 1e-15)

        # Compute log-loss for each theta
        costs = np.array([log_loss(y, expect[th]) for th in range(len(xs))])

        opt_idx = np.argmin(costs)
        return xs[opt_idx], costs[opt_idx]

    def reconstruct_circuit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: ParameterVector
    ) -> Tuple[QuantumCircuit, List[Dict], List[float]]:
        """
        Main training loop: iteratively reconstruct the circuit gate by gate.

        Args:
            X: Training features (N, n_features), already preprocessed to [0, π]
            y: Training labels (N,), values 0 or 1
            params: Parameter vector for data encoding

        Returns:
            (circuit, gate_sequence, cost_history)
        """
        self.circuit = self._initialize_circuit()
        self.gate_sequence = []
        self.cost_history = []

        n_features = X.shape[1] if X.ndim > 1 else 1
        count = 0

        while count < self.max_gates:
            t0 = time.time()

            best_cost = np.inf
            best_gate_idx = -1
            best_feature = -1
            best_theta = 0
            best_gate_desc = None

            # Evaluate each candidate gate
            for mf in range(len(self.gate_pool)):
                opt_thetas, opt_costs = self._evaluate_gate_candidate(
                    self.circuit, mf, X, y, params, count
                )

                for feat in range(n_features):
                    if self.verbose:
                        print(f'  Gate {self.gate_pool[mf]} Feature {feat} '
                              f'Angle {opt_thetas[feat]:.5f} Cost {opt_costs[feat]:.5f}',
                              flush=True)

                    if opt_costs[feat] < best_cost:
                        best_cost = opt_costs[feat]
                        best_gate_idx = mf
                        best_feature = feat
                        best_theta = opt_thetas[feat]
                        best_gate_desc = self.gate_pool[mf].copy()

            # Check convergence
            if count > 0 and len(self.cost_history) > 0:
                improvement = self.cost_history[-1] - best_cost
                if improvement < self.convergence_threshold:
                    if self.verbose:
                        print(f'Converged at gate {count}: improvement {improvement:.2e}')
                    break

            # Add the best gate to the circuit
            gate_desc_feat = best_gate_desc.copy()
            for q in range(self.n_qubits):
                if gate_desc_feat[q] != '111111':
                    g, _ = str(gate_desc_feat[q]).split('_')
                    gate_desc_feat[q] = f'{g}_{best_feature}'

            new_gate = build_gate_circuit(
                gate_desc_feat, self.n_qubits, best_theta,
                params, best_feature
            )
            self.circuit.compose(new_gate, qubits=list(range(self.n_qubits)), inplace=True)

            self.gate_sequence.append({
                'gate_descriptor': gate_desc_feat.tolist(),
                'angle': best_theta,
                'feature': best_feature,
                'gate_idx': best_gate_idx
            })
            self.cost_history.append(best_cost)

            t1 = time.time()
            if self.verbose:
                print(f'\n=== Gate {count}: cost={best_cost:.6f}, '
                      f'θ={best_theta:.5f}, feat={best_feature}, '
                      f'time={t1-t0:.1f}s ===\n', flush=True)

            count += 1

        return self.circuit, self.gate_sequence, self.cost_history

    def predict(
        self,
        X: np.ndarray,
        params: ParameterVector
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using the reconstructed circuit.

        Args:
            X: Input features
            params: Parameter vector

        Returns:
            (predictions, probabilities): Labels and raw probabilities
        """
        probs = []
        for xi in X:
            p = self._evaluate_expectation(self.circuit, xi, params)
            probs.append(p)

        probs = np.array(probs)
        # Map probability to expectation: expect = -1 + 2*prob
        expectations = -1 + 2 * probs
        predictions = np.where(expectations >= 0, 1, 0)

        return predictions, probs

    def get_measurement_count(self) -> int:
        """Get total measurements used."""
        return self.total_measurements

    def reset_measurements(self) -> None:
        """Reset measurement counter."""
        self.total_measurements = 0
