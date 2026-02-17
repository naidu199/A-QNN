"""
QNN Comparison Pipeline
========================

Provides a comprehensive comparison between:
1. Adaptive QNN (ARC) - Analytic Iterative Circuit Reconstruction
2. Fixed Ansatz QNN (COBYLA) - Standard VQC with gradient-free optimization
3. Fixed Ansatz QNN (SPSA) - Standard VQC with SPSA

Can run on both simulator and real IBM Quantum hardware.
Outputs detailed metrics, cost curves, and comparison tables.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn import preprocessing
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss, roc_auc_score,
    confusion_matrix
)
from qiskit.circuit import ParameterVector
import time
import json
import os


class QNNComparisonPipeline:
    """
    End-to-end comparison between ARC (dynamic) and Fixed Ansatz (static) QNNs.

    Usage:
        pipeline = QNNComparisonPipeline(n_qubits=4)
        results = pipeline.run_comparison(X_train, y_train, X_test, y_test)
        pipeline.print_report(results)
        pipeline.save_results(results, 'comparison_results.json')
    """

    def __init__(
        self,
        n_qubits: int = 4,
        arc_gate_list: List[str] = None,
        arc_max_gates: int = 50,
        arc_num_samples: int = 500,
        fixed_layers: int = 3,
        fixed_max_iter: int = 100,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            n_qubits: Number of qubits for all models
            arc_gate_list: Gate types for ARC method
            arc_max_gates: Max gates for ARC circuit building
            arc_num_samples: Classical samples for ARC theta optimization
            fixed_layers: Number of layers for fixed ansatz
            fixed_max_iter: Max iterations for fixed ansatz optimizer
            seed: Random seed
            verbose: Print progress
        """
        self.n_qubits = n_qubits
        self.arc_gate_list = arc_gate_list or ['U1', 'U2', 'U3', 'H', 'X', 'Z']
        self.arc_max_gates = arc_max_gates
        self.arc_num_samples = arc_num_samples
        self.fixed_layers = fixed_layers
        self.fixed_max_iter = fixed_max_iter
        self.seed = seed
        self.verbose = verbose

        np.random.seed(seed)

    def preprocess_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        scale_range: Tuple[float, float] = (0, np.pi)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data: scale features to [0, π] and ensure labels are 0/1.

        Args:
            X_train, X_test: Feature arrays
            y_train, y_test: Label arrays
            scale_range: Feature scaling range

        Returns:
            Preprocessed (X_train, X_test, y_train, y_test)
        """
        scaler = preprocessing.MinMaxScaler(
            feature_range=scale_range, clip=True
        )
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ensure labels are 0/1
        y_train_proc = np.where(y_train <= 0, 0, 1).astype(int)
        y_test_proc = np.where(y_test <= 0, 0, 1).astype(int)

        return X_train_scaled, X_test_scaled, y_train_proc, y_test_proc

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probs: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Per-class accuracy
        for c in np.unique(y_true):
            mask = y_true == c
            metrics[f'acc_class_{c}'] = np.mean(y_pred[mask] == c) if mask.any() else 0

        if probs is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, np.clip(probs, 1e-15, 1-1e-15))
            except Exception:
                metrics['log_loss'] = float('inf')
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, probs)
            except Exception:
                pass

        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    def train_arc_qnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the Adaptive QNN using Analytic Iterative Circuit Reconstruction.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Results dictionary
        """
        from ..estimators.arc_estimator import ARCEstimator

        n_features = X_train.shape[1]
        params = ParameterVector('x', n_features)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Training: ARC QNN (Analytic Iterative Reconstruction)")
            print("=" * 60)

        estimator = ARCEstimator(
            n_qubits=self.n_qubits,
            gate_list=self.arc_gate_list,
            num_samples_classical=self.arc_num_samples,
            max_gates=self.arc_max_gates,
            verbose=self.verbose
        )

        t0 = time.time()
        circuit, gate_sequence, cost_history = estimator.reconstruct_circuit(
            X_train, y_train, params
        )
        training_time = time.time() - t0

        # Predict
        train_preds, train_probs = estimator.predict(X_train, params)
        test_preds, test_probs = estimator.predict(X_test, params)

        # Metrics
        train_metrics = self._compute_metrics(y_train, train_preds, train_probs)
        test_metrics = self._compute_metrics(y_test, test_preds, test_probs)

        result = {
            'method': 'ARC_QNN',
            'training_time': training_time,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cost_history': cost_history,
            'n_gates': len(gate_sequence),
            'circuit_depth': circuit.depth(),
            'n_parameters': len(gate_sequence),  # Each gate has one trainable angle
            'total_measurements': estimator.get_measurement_count(),
            'gate_sequence': [
                {'gate': str(g['gate_descriptor']), 'angle': g['angle'], 'feature': g['feature']}
                for g in gate_sequence
            ],
            'circuit': circuit,
            'data_params': params,
            'estimator': estimator,
        }

        if self.verbose:
            print(f"\n  ARC QNN Results:")
            print(f"    Train accuracy: {train_metrics['accuracy']:.4f}")
            print(f"    Test accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"    Test balanced:  {test_metrics['balanced_accuracy']:.4f}")
            print(f"    Gates used:     {len(gate_sequence)}")
            print(f"    Circuit depth:  {circuit.depth()}")
            print(f"    Training time:  {training_time:.1f}s")

        return result

    def train_fixed_qnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        optimizer: str = 'cobyla'
    ) -> Dict[str, Any]:
        """
        Train a Fixed Ansatz QNN with the specified optimizer.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            optimizer: 'cobyla', 'nelder-mead', 'powell', or 'spsa'

        Returns:
            Results dictionary
        """
        from ..models.fixed_ansatz_qnn import FixedAnsatzQNN

        method_name = f'Fixed_QNN_{optimizer.upper()}'

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Training: {method_name}")
            print("=" * 60)

        model = FixedAnsatzQNN(
            n_qubits=self.n_qubits,
            n_layers=self.fixed_layers,
            optimizer=optimizer,
            max_iter=self.fixed_max_iter,
            verbose=self.verbose
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - t0

        # Predict
        train_preds, train_probs = model.predict(X_train)
        test_preds, test_probs = model.predict(X_test)

        # Metrics
        train_metrics = self._compute_metrics(y_train, train_preds, train_probs)
        test_metrics = self._compute_metrics(y_test, test_preds, test_probs)

        circuit_info = model.get_circuit_info()

        result = {
            'method': method_name,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cost_history': model.cost_history,
            'n_gates': circuit_info['depth'],
            'circuit_depth': circuit_info['depth'],
            'n_parameters': circuit_info['n_parameters'],
            'total_measurements': model.get_measurement_count(),
            'circuit': model.circuit,
            'data_params': model.data_params,
            'var_params': {p.name: float(v) for p, v in zip(model.var_params, model.optimal_params)}
                if model.optimal_params is not None else {},
            'model': model,
        }

        if self.verbose:
            print(f"\n  {method_name} Results:")
            print(f"    Train accuracy: {train_metrics['accuracy']:.4f}")
            print(f"    Test accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"    Test balanced:  {test_metrics['balanced_accuracy']:.4f}")
            print(f"    Parameters:     {circuit_info['n_parameters']}")
            print(f"    Circuit depth:  {circuit_info['depth']}")
            print(f"    Training time:  {training_time:.1f}s")

        return result

    def run_comparison(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        methods: List[str] = None,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Run full comparison between ARC and fixed ansatz QNNs.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            methods: Methods to compare (default: ['arc', 'cobyla', 'spsa'])
            preprocess: Whether to preprocess data

        Returns:
            Complete comparison results
        """
        if methods is None:
            methods = ['arc', 'cobyla']

        # Preprocess
        if preprocess:
            X_train, X_test, y_train, y_test = self.preprocess_data(
                X_train, X_test, y_train, y_test
            )

        if self.verbose:
            print(f"\nDataset: {len(X_train)} train, {len(X_test)} test, "
                  f"{X_train.shape[1]} features, {self.n_qubits} qubits")

        results = {
            'dataset_info': {
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': X_train.shape[1],
                'n_qubits': self.n_qubits,
                'class_distribution_train': {
                    str(c): int(np.sum(y_train == c)) for c in np.unique(y_train)
                },
            },
            'methods': {}
        }

        for method in methods:
            if method == 'arc':
                result = self.train_arc_qnn(X_train, y_train, X_test, y_test)
            elif method in ['cobyla', 'nelder-mead', 'powell', 'spsa']:
                result = self.train_fixed_qnn(X_train, y_train, X_test, y_test, method)
            else:
                print(f"Unknown method: {method}")
                continue

            results['methods'][result['method']] = result

        return results

    def run_on_hardware(
        self,
        results: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        backend_name: str = 'aer_simulator',
        shots: int = 4096,
        ibm_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run trained circuits on IBM Quantum hardware or simulator.

        Args:
            results: Results from run_comparison()
            X_test: Test data
            y_test: Test labels
            backend_name: IBM backend name
            shots: Number of shots
            ibm_token: API token

        Returns:
            Hardware execution results
        """
        from ..evaluation.ibm_runner import IBMQuantumRunner

        runner = IBMQuantumRunner(
            backend_name=backend_name,
            shots=shots,
            ibm_token=ibm_token,
            verbose=self.verbose
        )

        hw_results = {}

        for method_name, method_result in results['methods'].items():
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Running {method_name} on {backend_name}")
                print(f"{'='*50}")

            var_params = method_result.get('var_params', None)

            hw_result = runner.run_and_evaluate(
                circuit=method_result['circuit'],
                X_test=X_test,
                y_test=y_test,
                data_params=method_result['data_params'],
                var_params=var_params
            )

            hw_results[method_name] = hw_result

        results['hardware_results'] = {
            'backend': backend_name,
            'shots': shots,
            'results': hw_results
        }

        return results

    def print_report(self, results: Dict[str, Any]) -> str:
        """
        Print and return a formatted comparison report.

        Args:
            results: Results from run_comparison()

        Returns:
            Report string
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("QNN COMPARISON REPORT")
        report.append("Adaptive QNN (ARC) vs Fixed Ansatz QNN")
        report.append("=" * 80)

        # Dataset info
        info = results['dataset_info']
        report.append(f"\nDataset: {info['n_train']} train, {info['n_test']} test, "
                      f"{info['n_features']} features, {info['n_qubits']} qubits")

        # Table header
        report.append("\n" + "-" * 80)
        report.append(f"{'Metric':<25} ", )
        header = f"{'Metric':<25}"
        for method_name in results['methods']:
            header += f" | {method_name:<20}"
        report[-1] = header
        report.append("-" * 80)

        # Rows
        metrics_to_show = [
            ('Test Accuracy', 'test_metrics', 'accuracy'),
            ('Test Balanced Acc', 'test_metrics', 'balanced_accuracy'),
            ('Test F1 Score', 'test_metrics', 'f1_score'),
            ('Test Log-Loss', 'test_metrics', 'log_loss'),
            ('Train Accuracy', 'train_metrics', 'accuracy'),
            ('Circuit Depth', None, 'circuit_depth'),
            ('Num Parameters', None, 'n_parameters'),
            ('Num Gates', None, 'n_gates'),
            ('Training Time (s)', None, 'training_time'),
            ('Total Measurements', None, 'total_measurements'),
        ]

        for label, sub_key, metric_key in metrics_to_show:
            row = f"{label:<25}"
            for method_name, method_result in results['methods'].items():
                if sub_key:
                    val = method_result.get(sub_key, {}).get(metric_key, 'N/A')
                else:
                    val = method_result.get(metric_key, 'N/A')

                if isinstance(val, float):
                    row += f" | {val:<20.4f}"
                else:
                    row += f" | {str(val):<20}"
            report.append(row)

        report.append("-" * 80)

        # Hardware results
        if 'hardware_results' in results:
            hw = results['hardware_results']
            report.append(f"\nHardware Results (Backend: {hw['backend']}, Shots: {hw['shots']})")
            report.append("-" * 80)

            hw_header = f"{'Metric':<25}"
            for method_name in hw['results']:
                hw_header += f" | {method_name:<20}"
            report.append(hw_header)
            report.append("-" * 80)

            hw_metrics = [
                ('HW Accuracy', 'accuracy'),
                ('HW Balanced Acc', 'balanced_accuracy'),
                ('HW Acc (Opt Thresh)', 'accuracy_optimal_threshold'),
                ('HW Log-Loss', 'log_loss'),
                ('HW Exec Time (s)', 'execution_time'),
            ]

            for label, key in hw_metrics:
                row = f"{label:<25}"
                for method_name, hw_result in hw['results'].items():
                    val = hw_result.get(key, 'N/A')
                    if isinstance(val, float):
                        row += f" | {val:<20.4f}"
                    else:
                        row += f" | {str(val):<20}"
                report.append(row)
            report.append("-" * 80)

        # Conclusion
        report.append("\nConclusion:")
        methods = results['methods']
        if len(methods) >= 2:
            method_names = list(methods.keys())
            accs = {m: methods[m]['test_metrics']['accuracy'] for m in method_names}
            best = max(accs, key=accs.get)
            report.append(f"  Best test accuracy: {best} ({accs[best]:.4f})")

            if 'ARC_QNN' in methods:
                arc_acc = methods['ARC_QNN']['test_metrics']['accuracy']
                for other in method_names:
                    if other != 'ARC_QNN':
                        other_acc = methods[other]['test_metrics']['accuracy']
                        diff = arc_acc - other_acc
                        report.append(f"  ARC vs {other}: {diff:+.4f} accuracy difference")

        report_str = "\n".join(report)
        print(report_str)
        return report_str

    def save_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ):
        """
        Save results to JSON file.

        Args:
            results: Results dictionary
            filepath: Output file path
        """
        # Make results JSON-serializable
        serializable = {
            'dataset_info': results['dataset_info'],
            'methods': {}
        }

        for method_name, method_result in results['methods'].items():
            serializable['methods'][method_name] = {
                'method': method_result['method'],
                'training_time': method_result['training_time'],
                'train_metrics': {
                    k: v for k, v in method_result['train_metrics'].items()
                    if not isinstance(v, np.ndarray)
                },
                'test_metrics': {
                    k: v for k, v in method_result['test_metrics'].items()
                    if not isinstance(v, np.ndarray)
                },
                'cost_history': [float(c) for c in method_result['cost_history']],
                'n_gates': method_result['n_gates'],
                'circuit_depth': method_result['circuit_depth'],
                'n_parameters': method_result['n_parameters'],
                'total_measurements': method_result['total_measurements'],
            }

        if 'hardware_results' in results:
            serializable['hardware_results'] = {
                'backend': results['hardware_results']['backend'],
                'shots': results['hardware_results']['shots'],
                'results': {}
            }
            for method_name, hw_result in results['hardware_results']['results'].items():
                serializable['hardware_results']['results'][method_name] = {
                    k: v for k, v in hw_result.items()
                    if isinstance(v, (int, float, str, list, dict, bool))
                }

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        if self.verbose:
            print(f"\nResults saved to: {filepath}")
