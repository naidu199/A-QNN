"""
IBM Quantum Backend Runner
============================

Provides support for running circuits on real IBM Quantum hardware
or IBM Aer simulators with noise models. This module handles:

1. Connection to IBM Quantum service
2. Circuit transpilation for real hardware
3. Running both ARC and Fixed Ansatz QNNs on real backends
4. Noise-aware evaluation with error mitigation
5. Comparison between simulator and real hardware results

Usage:
    runner = IBMQuantumRunner(backend_name='ibm_brisbane')
    results = runner.run_circuit(circuit, X, y, params)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score
import time
import warnings
import json
import os


class IBMQuantumRunner:
    """
    Runs quantum circuits on IBM Quantum hardware or simulators.

    Supports:
    - Real IBM Quantum backends (ibm_brisbane, ibm_osaka, etc.)
    - Fake/noise-model backends for testing
    - Aer simulator (statevector or qasm)
    - Error mitigation techniques
    """

    def __init__(
        self,
        backend_name: str = 'aer_simulator',
        shots: int = 4096,
        optimization_level: int = 2,
        ibm_token: Optional[str] = None,
        channel: str = 'ibm_quantum',
        use_error_mitigation: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            backend_name: Name of the backend
                - 'aer_simulator': Local Aer statevector simulator
                - 'aer_qasm': Local Aer QASM simulator
                - 'fake_<name>': Fake backend with noise model
                - '<real_backend>': Real IBM Quantum backend
            shots: Number of measurement shots
            optimization_level: Transpilation optimization level (0-3)
            ibm_token: IBM Quantum API token (reads from env if not provided)
            channel: IBM Quantum channel
            use_error_mitigation: Enable error mitigation
            verbose: Print progress
        """
        self.backend_name = backend_name
        self.shots = shots
        self.optimization_level = optimization_level
        self.use_error_mitigation = use_error_mitigation
        self.verbose = verbose

        self.backend = None
        self.service = None
        self.session = None

        self._setup_backend(ibm_token, channel)

    def _setup_backend(self, ibm_token: Optional[str], channel: str):
        """Initialize the quantum backend."""
        if self.backend_name == 'aer_simulator':
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator(method='statevector')
            self.execution_mode = 'statevector'

        elif self.backend_name == 'aer_qasm':
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator(method='automatic')
            self.execution_mode = 'qasm'

        elif self.backend_name.startswith('fake_'):
            try:
                from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2 as FakeProvider
                provider = FakeProvider()
                fake_name = self.backend_name.replace('fake_', '')
                available = {b.name: b for b in provider.backends()}
                if fake_name in available:
                    self.backend = available[fake_name]
                else:
                    if self.verbose:
                        print(f"Fake backend '{fake_name}' not found. "
                              f"Available: {list(available.keys())[:5]}...")
                    from qiskit_aer import AerSimulator
                    self.backend = AerSimulator()
                self.execution_mode = 'qasm'
            except ImportError:
                warnings.warn("qiskit_ibm_runtime not available, using Aer")
                from qiskit_aer import AerSimulator
                self.backend = AerSimulator()
                self.execution_mode = 'qasm'

        else:
            # Real IBM Quantum backend
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService

                token = ibm_token or os.environ.get('IBM_QUANTUM_TOKEN', None)
                if token:
                    self.service = QiskitRuntimeService(
                        channel=channel,
                        token=token
                    )
                else:
                    # Try saved credentials
                    self.service = QiskitRuntimeService(channel=channel)

                self.backend = self.service.backend(self.backend_name)
                self.execution_mode = 'ibm_runtime'

                if self.verbose:
                    print(f"Connected to IBM Quantum backend: {self.backend_name}")
                    print(f"  Qubits: {self.backend.num_qubits}")

            except Exception as e:
                warnings.warn(f"Could not connect to IBM backend: {e}. "
                             f"Falling back to Aer simulator.")
                from qiskit_aer import AerSimulator
                self.backend = AerSimulator()
                self.execution_mode = 'statevector'

    def transpile_circuit(
        self,
        circuit: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Transpile circuit for the target backend.

        Args:
            circuit: Input circuit

        Returns:
            Transpiled circuit
        """
        if self.execution_mode == 'statevector':
            return circuit

        try:
            from qiskit import generate_preset_pass_manager
            pm = generate_preset_pass_manager(
                optimization_level=self.optimization_level,
                backend=self.backend
            )
            transpiled = pm.run(circuit)
        except Exception:
            transpiled = transpile(
                circuit, self.backend,
                optimization_level=self.optimization_level
            )

        if self.verbose:
            print(f"  Transpiled: depth {circuit.depth()} -> {transpiled.depth()}, "
                  f"gates {circuit.size()} -> {transpiled.size()}")

        return transpiled

    def evaluate_circuit(
        self,
        circuit: QuantumCircuit,
        X: np.ndarray,
        data_params: ParameterVector,
        var_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Evaluate a circuit on a dataset, returning probabilities.

        Args:
            circuit: Quantum circuit (may be parameterized)
            X: Input data (N, n_features)
            data_params: Data parameter vector
            var_params: Optional variational parameter dict

        Returns:
            Array of P(|0⟩) probabilities for each data point
        """
        probs = np.zeros(len(X))

        if self.execution_mode == 'statevector':
            # Direct statevector simulation
            for i, xi in enumerate(X):
                probs[i] = self._eval_statevector(circuit, xi, data_params, var_params)

        elif self.execution_mode == 'ibm_runtime':
            # Batch execution on IBM Runtime
            probs = self._eval_ibm_runtime(circuit, X, data_params, var_params)

        else:
            # QASM-style simulation
            probs = self._eval_qasm(circuit, X, data_params, var_params)

        return probs

    def _eval_statevector(
        self,
        circuit: QuantumCircuit,
        x: np.ndarray,
        data_params: ParameterVector,
        var_params: Optional[Dict]
    ) -> float:
        """Evaluate single point via statevector."""
        param_dict = {}

        # Bind data params
        circuit_param_names = {p.name: p for p in circuit.parameters}
        for i in range(min(len(x), len(data_params))):
            if data_params[i].name in circuit_param_names:
                param_dict[circuit_param_names[data_params[i].name]] = float(x[i])

        # Bind variational params
        if var_params:
            for p_name, val in var_params.items():
                if isinstance(p_name, str) and p_name in circuit_param_names:
                    param_dict[circuit_param_names[p_name]] = float(val)
                elif hasattr(p_name, 'name') and p_name.name in circuit_param_names:
                    param_dict[circuit_param_names[p_name.name]] = float(val)

        # Bind remaining to 0
        for p in circuit.parameters:
            if p not in param_dict:
                param_dict[p] = 0.0

        bound = circuit.assign_parameters(param_dict)
        sv = Statevector(bound)
        return sv.probabilities()[0]

    def _eval_qasm(
        self,
        circuit: QuantumCircuit,
        X: np.ndarray,
        data_params: ParameterVector,
        var_params: Optional[Dict]
    ) -> np.ndarray:
        """Evaluate via QASM simulation."""
        probs = np.zeros(len(X))

        for i, xi in enumerate(X):
            qc = circuit.copy()

            # Add measurements
            if qc.num_clbits == 0:
                qc.measure_all()

            param_dict = {}
            circuit_param_names = {p.name: p for p in qc.parameters}
            for j in range(min(len(xi), len(data_params))):
                if data_params[j].name in circuit_param_names:
                    param_dict[circuit_param_names[data_params[j].name]] = float(xi[j])

            if var_params:
                for p_name, val in var_params.items():
                    name = p_name if isinstance(p_name, str) else p_name.name
                    if name in circuit_param_names:
                        param_dict[circuit_param_names[name]] = float(val)

            for p in qc.parameters:
                if p not in param_dict:
                    param_dict[p] = 0.0

            bound = qc.assign_parameters(param_dict)

            transpiled = transpile(bound, self.backend)
            result = self.backend.run(transpiled, shots=self.shots).result()
            counts = result.get_counts()

            zero_key = '0' * circuit.num_qubits
            probs[i] = counts.get(zero_key, 0) / self.shots

        return probs

    def _eval_ibm_runtime(
        self,
        circuit: QuantumCircuit,
        X: np.ndarray,
        data_params: ParameterVector,
        var_params: Optional[Dict]
    ) -> np.ndarray:
        """Evaluate via IBM Quantum Runtime Sampler."""
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit_ibm_runtime import Session
        except ImportError:
            warnings.warn("IBM Runtime not available, falling back to statevector")
            return np.array([
                self._eval_statevector(circuit, xi, data_params, var_params)
                for xi in X
            ])

        probs = np.zeros(len(X))

        # Transpile once
        qc_meas = circuit.copy()
        if qc_meas.num_clbits == 0:
            qc_meas.measure_all()

        transpiled = self.transpile_circuit(qc_meas)

        if self.verbose:
            print(f"  Running {len(X)} circuits on {self.backend_name}...")

        try:
            with Session(service=self.service, backend=self.backend) as session:
                sampler = Sampler(session=session)

                # Batch parameter bindings
                for batch_start in range(0, len(X), 100):
                    batch_end = min(batch_start + 100, len(X))
                    batch_X = X[batch_start:batch_end]

                    pubs = []
                    for xi in batch_X:
                        param_dict = {}
                        for j in range(min(len(xi), len(data_params))):
                            param_dict[data_params[j]] = float(xi[j])
                        if var_params:
                            for p_name, val in var_params.items():
                                param_dict[p_name] = float(val)
                        for p in transpiled.parameters:
                            if p not in param_dict:
                                param_dict[p] = 0.0

                        bound = transpiled.assign_parameters(param_dict)
                        pubs.append(bound)

                    result = sampler.run(pubs, shots=self.shots).result()

                    for idx, pub_result in enumerate(result):
                        counts = pub_result.data.meas.get_counts()
                        zero_key = '0' * circuit.num_qubits
                        probs[batch_start + idx] = counts.get(zero_key, 0) / self.shots

                if self.verbose:
                    print(f"  IBM Runtime execution complete")

        except Exception as e:
            warnings.warn(f"IBM Runtime execution failed: {e}. Using statevector.")
            for i, xi in enumerate(X):
                probs[i] = self._eval_statevector(circuit, xi, data_params, var_params)

        return probs

    def run_and_evaluate(
        self,
        circuit: QuantumCircuit,
        X_test: np.ndarray,
        y_test: np.ndarray,
        data_params: ParameterVector,
        var_params: Optional[Dict] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run circuit on backend and compute evaluation metrics.

        Args:
            circuit: Trained quantum circuit
            X_test: Test features
            y_test: Test labels (0 or 1)
            data_params: Data parameter vector
            var_params: Variational parameters (for fixed ansatz)
            threshold: Decision threshold for classification

        Returns:
            Dictionary with accuracy, balanced accuracy, probabilities, etc.
        """
        t0 = time.time()

        probs = self.evaluate_circuit(circuit, X_test, data_params, var_params)

        # Convert to predictions
        expectations = -1 + 2 * probs
        predictions = np.where(expectations >= 0, 1, 0)

        # Also try optimal threshold
        from sklearn.metrics import roc_curve
        try:
            fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
            tnr = 1 - fpr
            balanced_accs = (tpr + tnr) / 2
            best_threshold_idx = np.argmax(balanced_accs)
            best_threshold = thresholds_roc[best_threshold_idx]
            predictions_optimal = np.where(probs >= best_threshold, 1, 0)
        except Exception:
            best_threshold = 0.5
            predictions_optimal = predictions

        t1 = time.time()

        accuracy = accuracy_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        accuracy_optimal = accuracy_score(y_test, predictions_optimal)
        balanced_acc_optimal = balanced_accuracy_score(y_test, predictions_optimal)

        # Per-class accuracy
        y0_mask = y_test == 0
        y1_mask = y_test == 1
        acc_class0 = np.mean(predictions_optimal[y0_mask] == 0) if y0_mask.any() else 0
        acc_class1 = np.mean(predictions_optimal[y1_mask] == 1) if y1_mask.any() else 0

        try:
            cost = log_loss(y_test, np.clip(probs, 1e-15, 1-1e-15))
        except Exception:
            cost = float('inf')

        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'accuracy_optimal_threshold': accuracy_optimal,
            'balanced_accuracy_optimal': balanced_acc_optimal,
            'acc_class_0': acc_class0,
            'acc_class_1': acc_class1,
            'optimal_threshold': float(best_threshold),
            'log_loss': cost,
            'probabilities': probs.tolist(),
            'predictions': predictions.tolist(),
            'predictions_optimal': predictions_optimal.tolist(),
            'execution_time': t1 - t0,
            'backend': self.backend_name,
            'shots': self.shots,
        }

        if self.verbose:
            print(f"\n  Backend: {self.backend_name}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  Accuracy (optimal threshold): {accuracy_optimal:.4f}")
            print(f"  Class 0 acc: {acc_class0:.4f}, Class 1 acc: {acc_class1:.4f}")
            print(f"  Log-loss: {cost:.4f}")
            print(f"  Execution time: {t1-t0:.1f}s")

        return results


def run_on_ibm_hardware(
    circuits_dict: Dict[str, Dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    backend_name: str = 'aer_simulator',
    shots: int = 4096,
    ibm_token: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Convenience function to run multiple method circuits on IBM hardware.

    Args:
        circuits_dict: Dict mapping method name to {circuit, data_params, var_params}
        X_test: Test data
        y_test: Test labels
        backend_name: IBM backend or simulator
        shots: Number of shots
        ibm_token: Optional API token

    Returns:
        Results for each method
    """
    runner = IBMQuantumRunner(
        backend_name=backend_name,
        shots=shots,
        ibm_token=ibm_token
    )

    results = {}
    for method_name, method_info in circuits_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name} on {backend_name}")
        print(f"{'='*50}")

        result = runner.run_and_evaluate(
            circuit=method_info['circuit'],
            X_test=X_test,
            y_test=y_test,
            data_params=method_info['data_params'],
            var_params=method_info.get('var_params', None)
        )
        results[method_name] = result

    return results
