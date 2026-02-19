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
from joblib import Parallel, delayed
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
        # Matches reference circ_convertSinge exactly:
        # U1=Rz, U2=Rx, U3=Ry, H=just H (no Rz), two-qubit gates use Rz after
        # Python negative indexing for wrap-around (j-1 at j=0 → last qubit)
        if gate_type == 'U1':
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'U2':
            circuit.rx(angle * param_val, qreg[j])
        elif gate_type == 'U3':
            circuit.ry(angle * param_val, qreg[j])
        elif gate_type == 'H':
            circuit.h(qreg[j])  # reference circ_convertSinge: H has no Rz
        elif gate_type == 'Rz':
            circuit.rz(angle, qreg[j])
        elif gate_type == 'Rx':
            circuit.rx(angle, qreg[j])
        elif gate_type == 'Ry':
            circuit.ry(angle, qreg[j])
        elif gate_type == 'P':
            circuit.p(angle * param_val, qreg[j])
        elif gate_type == 'CP':
            circuit.cp(angle * param_val, qreg[j-1], qreg[j])
        elif gate_type == 'X':
            circuit.cx(qreg[j-1], qreg[j])
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'Xn':
            circuit.cx(qreg[j-2], qreg[j])
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'Z':
            circuit.cz(qreg[j-1], qreg[j])
            circuit.rz(angle * param_val, qreg[j])
        elif gate_type == 'Zn':
            circuit.cz(qreg[j-2], qreg[j])
            circuit.rz(angle * param_val, qreg[j])

    return circuit


def _compute_abc_for_sample(base_circ_bound, test_gates_3, n_qubits):
    """Worker for parallel per-sample ABC computation (module-level for pickle)."""
    qubits = list(range(n_qubits))
    expect = [0.0, 0.0, 0.0]
    for kt in range(3):
        tc = base_circ_bound.copy()
        tc.compose(test_gates_3[kt], qubits=qubits, inplace=True)
        rem = {p: 0.0 for p in tc.parameters}
        if rem:
            tc = tc.assign_parameters(rem)
        expect[kt] = Statevector(tc).probabilities()[0]
    return determine_sine_curve(expect[0], expect[1], expect[2])


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
        verbose: bool = True,
        subsample_size: int = 100,
        n_jobs: int = -1,
        patience: int = 1
    ):
        """
        Args:
            n_qubits: Number of qubits
            gate_list: Gate types to use (default: ['U1', 'U2', 'U3', 'H', 'X', 'Z'])
            num_samples_classical: Number of theta samples for classical optimization
            convergence_threshold: Stop when cost improvement < this
            max_gates: Maximum number of gates to add
            verbose: Print progress
            subsample_size: Training samples per gate iteration (reference uses 100)
            n_jobs: Number of parallel jobs for joblib (-1 = all cores)
            patience: Number of consecutive non-improving gates to tolerate before stopping
                      (1 = stop after 1 non-improving gate (original behavior),
                       2 = allow 1 extra chance, etc.)
        """
        self.n_qubits = n_qubits
        self.gate_list = gate_list or ['U1', 'U2', 'U3', 'H', 'X', 'Z']
        self.num_samples_classical = num_samples_classical
        self.convergence_threshold = convergence_threshold
        self.max_gates = max_gates
        self.verbose = verbose
        self.subsample_size = subsample_size
        self.n_jobs = n_jobs
        self.patience = patience

        # Build gate pool
        self.gate_pool = ARCGatePool(self.gate_list, n_qubits)

        # Training state
        self.circuit = None
        self.gate_sequence = []  # List of (gate_descriptor, angle, feature)
        self.cost_history = []
        self.total_measurements = 0

    def _initialize_circuit(self) -> QuantumCircuit:
        """Create empty initial circuit (matching reference: initializeMainCirc)."""
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
        gate_desc: np.ndarray,
        test_gates_3: List[QuantumCircuit],
        base_circuits_bound: Optional[List[QuantumCircuit]],
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
        count: int,
        n_samples: int
    ) -> Tuple[List[float], List[float]]:
        """
        Evaluate a candidate gate across all features.

        KEY OPTIMIZATION (matching reference impl):
        ABC values are feature-independent because the test gate parameter
        is always set to 1.0. We compute ABC once per sample, then search
        over features only in the fast numpy-based theta optimization.
        This is exactly how the reference layerParallel + find_cost_rec_QNN work.

        Args:
            gate_desc: Gate descriptor array
            test_gates_3: Pre-built test gates for theta=0, pi/2, -pi/2 (fully bound)
            base_circuits_bound: Pre-bound base circuits per sample (None for count=0)
            X: Training data
            y: Training labels
            n_features: Number of features (incl. bias)
            count: Current gate count (0 = first gate)
            n_samples: Number of training samples

        Returns:
            (optimal_thetas, optimal_costs) for each feature
        """
        AA = np.zeros(n_samples)
        BB = np.zeros(n_samples)
        CC = np.zeros(n_samples)
        qubits = list(range(self.n_qubits))

        if count == 0:
            # Empty base circuit: ABC identical for all samples (only 3 sims)
            expect = [0.0, 0.0, 0.0]
            for kt in range(3):
                sv = Statevector(test_gates_3[kt])
                expect[kt] = sv.probabilities()[0]
            a, b, c = determine_sine_curve(expect[0], expect[1], expect[2])
            AA[:] = a
            BB[:] = b
            CC[:] = c
        else:
            # Data-dependent ABC per sample — parallel across samples
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_abc_for_sample)(
                    base_circuits_bound[i], test_gates_3, self.n_qubits
                ) for i in range(n_samples)
            )
            for i, (a, b, c) in enumerate(results):
                AA[i] = a
                BB[i] = b
                CC[i] = c

        # Search optimal theta per feature (pure numpy, very fast)
        opt_thetas = []
        opt_costs = []
        for feat in range(n_features):
            t, cost = self._find_optimal_theta(AA, BB, CC, X, y, feat)
            opt_thetas.append(t)
            opt_costs.append(cost)

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

        # Vectorized binary cross-entropy (replaces 1001 sklearn.log_loss calls)
        costs = -(y[None, :] * np.log(expect)
                  + (1 - y[None, :]) * np.log(1 - expect)).mean(axis=1)

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
        # ---- Add constant bias feature (π) like reference paper ----
        # Reference: normalized_Xtrain[i][-1] = np.pi
        # Ensures at least one feature always has full rotation range
        bias = np.full((len(X), 1), np.pi)
        X = np.hstack([X, bias])
        n_features = X.shape[1]
        if len(params) < n_features:
            params = ParameterVector('x', n_features)
        if self.verbose:
            print(f"  Bias feature (pi) added -> {n_features} total features")

        self.circuit = self._initialize_circuit()
        self.gate_sequence = []
        self.cost_history = []
        self._params = params
        self._X_has_bias = True
        count = 0
        no_improve_count = 0
        best_overall_cost = np.inf
        best_overall_gate_count = 0
        best_overall_circuit = None
        best_overall_gate_sequence = []
        best_overall_cost_history = []
        n_pool = len(self.gate_pool)
        qubits_list = list(range(self.n_qubits))
        theta_test = [0, np.pi * 0.5, -np.pi * 0.5]

        if self.verbose:
            print(f"  Gate pool: {n_pool} candidates, "
                  f"{n_features} features, n_jobs={self.n_jobs}")

        while count < self.max_gates:
            t0 = time.time()

            # Subsample training data for speed
            if self.subsample_size > 0 and len(X) > self.subsample_size:
                idx = np.random.choice(len(X), self.subsample_size, replace=False)
                X_sub, y_sub = X[idx], y[idx]
            else:
                X_sub, y_sub = X, y
            n_samp = len(X_sub)

            # --- Pre-bind base circuits (REUSED by all gate candidates) ---
            if count > 0:
                param_name_map = {params[fi].name: fi for fi in range(n_features)}
                base_circuits_bound = []
                for i in range(n_samp):
                    tc = self.circuit.copy()
                    bind = {}
                    for p in tc.parameters:
                        fi = param_name_map.get(p.name)
                        bind[p] = float(X_sub[i][fi]) if fi is not None else 0.0
                    tc = tc.assign_parameters(bind)
                    base_circuits_bound.append(tc)
            else:
                base_circuits_bound = None

            # --- Pre-build test gates for ALL candidates ---
            # ABC is feature-independent (test param=1.0), so feature=0 is used
            # for building; the actual feature search happens in theta optimization.
            all_test_gates = []
            for mf in range(n_pool):
                gate_desc = self.gate_pool[mf]
                tg3 = []
                for kt in range(3):
                    gd = gate_desc.copy()
                    for q in range(self.n_qubits):
                        if gd[q] != '111111':
                            g, _ = str(gd[q]).split('_')
                            gd[q] = f'{g}_0'
                    tg = build_gate_circuit(
                        gd, self.n_qubits, theta_test[kt],
                        ParameterVector('t', n_features), 0
                    )
                    tp = {p: 1.0 for p in tg.parameters}
                    if tp:
                        tg = tg.assign_parameters(tp)
                    tg3.append(tg)
                all_test_gates.append(tg3)

            t_prep = time.time() - t0

            best_cost = np.inf
            best_gate_idx = -1
            best_feature = -1
            best_theta = 0
            best_gate_desc = None

            # --- Evaluate gate candidates sequentially ---
            # (per-sample parallelism inside _evaluate_gate_candidate for count>0)
            for mf in range(n_pool):
                opt_thetas, opt_costs = self._evaluate_gate_candidate(
                    self.gate_pool[mf], all_test_gates[mf],
                    base_circuits_bound, X_sub, y_sub,
                    n_features, count, n_samp
                )
                for feat in range(n_features):
                    if opt_costs[feat] < best_cost:
                        best_cost = opt_costs[feat]
                        best_gate_idx = mf
                        best_feature = feat
                        best_theta = opt_thetas[feat]
                        best_gate_desc = self.gate_pool[mf].copy()

            # Update measurement count
            if count == 0:
                self.total_measurements += n_pool * 3
            else:
                self.total_measurements += n_pool * n_samp * 3

            # --- Add the best gate ---
            gate_desc_feat = best_gate_desc.copy()
            for q in range(self.n_qubits):
                if gate_desc_feat[q] != '111111':
                    g, _ = str(gate_desc_feat[q]).split('_')
                    gate_desc_feat[q] = f'{g}_{best_feature}'

            new_gate = build_gate_circuit(
                gate_desc_feat, self.n_qubits, best_theta,
                params, best_feature
            )
            self.circuit.compose(new_gate, qubits=qubits_list, inplace=True)

            self.gate_sequence.append({
                'gate_descriptor': gate_desc_feat.tolist(),
                'angle': best_theta,
                'feature': best_feature,
                'gate_idx': best_gate_idx
            })
            self.cost_history.append(best_cost)
            count += 1

            t1 = time.time()
            if self.verbose:
                gate_name = best_gate_desc.tolist()
                print(f'Gate {count}: cost={best_cost:.6f}, '
                      f'theta={best_theta:.5f}, feat={best_feature}, '
                      f'gate={gate_name}, '
                      f'samples={n_samp}, '
                      f'prep={t_prep:.1f}s eval={t1-t0-t_prep:.1f}s '
                      f'total={t1-t0:.1f}s', flush=True)

            # Convergence check with patience
            if count > 1 and len(self.cost_history) >= 2:
                if self.cost_history[-2] - self.cost_history[-1] < self.convergence_threshold:
                    no_improve_count = no_improve_count + 1
                    if no_improve_count >= self.patience:
                        if self.verbose:
                            print(f'Converged at gate {count} (patience={self.patience})')
                        # Revert to best circuit if we overshot
                        if best_overall_gate_count < count:
                            self.circuit = best_overall_circuit
                            self.gate_sequence = best_overall_gate_sequence
                            self.cost_history = best_overall_cost_history
                            if self.verbose:
                                print(f'  Reverted to best circuit at gate {best_overall_gate_count} '
                                      f'(cost={best_overall_cost:.6f})')
                        break
                else:
                    no_improve_count = 0

            # Track best circuit seen so far
            if best_cost < best_overall_cost:
                best_overall_cost = best_cost
                best_overall_gate_count = count
                best_overall_circuit = self.circuit.copy()
                best_overall_gate_sequence = list(self.gate_sequence)
                best_overall_cost_history = list(self.cost_history)

        # Final: ensure we return the best circuit found
        if best_overall_gate_count < count:
            self.circuit = best_overall_circuit
            self.gate_sequence = best_overall_gate_sequence
            self.cost_history = best_overall_cost_history
            if self.verbose:
                print(f'Using best circuit from gate {best_overall_gate_count} '
                      f'(cost={best_overall_cost:.6f})')

        return self.circuit, self.gate_sequence, self.cost_history

    def predict(
        self,
        X: np.ndarray,
        params: ParameterVector = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using the reconstructed circuit.

        Args:
            X: Input features
            params: Parameter vector (uses stored params if None)

        Returns:
            (predictions, probabilities): Labels and raw probabilities
        """
        if params is None:
            params = self._params

        # Append bias feature (always added during training)
        if getattr(self, '_X_has_bias', False):
            bias = np.full((len(X), 1), np.pi)
            X = np.hstack([X, bias])

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
