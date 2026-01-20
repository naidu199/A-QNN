"""
QNN Trainer
============

Comprehensive training utilities for Adaptive Quantum Neural Networks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

from ..models.adaptive_qnn import AdaptiveQNN


@dataclass
class TrainingConfig:
    """Configuration for QNN training."""
    max_iterations: int = 20
    improvement_threshold: float = 1e-4
    patience: int = 5
    validation_split: float = 0.2
    batch_size: Optional[int] = None
    shuffle: bool = True
    random_state: Optional[int] = None
    verbose: bool = True
    checkpoint_dir: Optional[str] = None
    log_interval: int = 1


@dataclass
class TrainingResult:
    """Results from training."""
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    train_acc_history: List[float] = field(default_factory=list)
    val_acc_history: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_val_acc: float = 0.0
    best_iteration: int = 0
    total_time: float = 0.0
    n_gates: int = 0
    n_parameters: int = 0
    total_measurements: int = 0


class QNNTrainer:
    """
    Trainer class for Adaptive Quantum Neural Networks.

    Provides comprehensive training functionality including:
    - Train/validation splitting
    - Early stopping
    - Checkpointing
    - Training history tracking
    - Multiple training strategies

    Example:
        >>> from src.training import QNNTrainer, TrainingConfig
        >>> from src.models import AdaptiveQNN
        >>>
        >>> config = TrainingConfig(
        ...     max_iterations=20,
        ...     patience=5,
        ...     validation_split=0.2
        ... )
        >>>
        >>> model = AdaptiveQNN(n_qubits=4, n_classes=2)
        >>> trainer = QNNTrainer(model, config)
        >>> result = trainer.fit(X_train, y_train)
    """

    def __init__(
        self,
        model: AdaptiveQNN,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: AdaptiveQNN model to train
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.result = TrainingResult()
        self.callbacks: List[Callable] = []

    def add_callback(self, callback: Callable) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets."""
        n_samples = len(X)
        n_val = int(n_samples * self.config.validation_split)

        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def _compute_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """Compute loss and accuracy."""
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)

        # Compute loss
        probs = self.model.predict_proba(X)
        loss = 0.0
        for i, yi in enumerate(y):
            loss -= np.log(probs[i, int(yi)] + 1e-10)
        loss /= len(y)

        return loss, accuracy

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingResult:
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            TrainingResult with training history
        """
        start_time = time.time()

        # Split data if validation set not provided
        if X_val is None and self.config.validation_split > 0:
            X_train, y_train, X_val, y_val = self._split_data(X, y)
        else:
            X_train, y_train = X, y

        if self.config.verbose:
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")

        # Train the model
        self.model.fit(
            X_train, y_train,
            max_iterations=self.config.max_iterations,
            improvement_threshold=self.config.improvement_threshold,
            verbose=self.config.verbose
        )

        # Get training history from model
        history = self.model.get_training_history()
        self.result.train_loss_history = [h['cost'] for h in history]

        # Compute final metrics
        train_loss, train_acc = self._compute_metrics(X_train, y_train)
        self.result.train_acc_history.append(train_acc)

        if X_val is not None:
            val_loss, val_acc = self._compute_metrics(X_val, y_val)
            self.result.val_loss_history.append(val_loss)
            self.result.val_acc_history.append(val_acc)
            self.result.best_val_loss = val_loss
            self.result.best_val_acc = val_acc

        # Record final state
        self.result.total_time = time.time() - start_time
        circuit_info = self.model.get_circuit_info()
        self.result.n_gates = circuit_info['n_gates']
        self.result.n_parameters = circuit_info['n_parameters']
        self.result.total_measurements = self.model.estimator.get_measurement_count()

        if self.config.verbose:
            print(f"\n=== Training Summary ===")
            print(f"Training time: {self.result.total_time:.2f}s")
            print(f"Training accuracy: {train_acc:.4f}")
            if X_val is not None:
                print(f"Validation accuracy: {val_acc:.4f}")
            print(f"Circuit gates: {self.result.n_gates}")
            print(f"Parameters: {self.result.n_parameters}")
            print(f"Total measurements: {self.result.total_measurements}")

        return self.result

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            X: Features
            y: Labels
            n_folds: Number of folds

        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=self.config.random_state)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            if self.config.verbose:
                print(f"\n=== Fold {fold + 1}/{n_folds} ===")

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Create fresh model for each fold
            from ..models.adaptive_qnn import create_adaptive_qnn
            model = create_adaptive_qnn(
                n_qubits=self.model.n_qubits,
                n_classes=self.model.n_classes,
                encoding_type=self.model.encoding_type,
                max_gates=self.model.max_gates,
                shots=self.model.shots
            )

            # Train
            model.fit(X_train, y_train,
                     max_iterations=self.config.max_iterations,
                     verbose=False)

            # Evaluate
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)

            fold_results.append({
                'fold': fold + 1,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'n_gates': len(model.circuit_builder.gate_set.gate_history),
                'n_params': len(model.trained_params)
            })

            if self.config.verbose:
                print(f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")

        # Aggregate results
        val_accs = [r['val_accuracy'] for r in fold_results]

        return {
            'fold_results': fold_results,
            'mean_val_accuracy': np.mean(val_accs),
            'std_val_accuracy': np.std(val_accs),
            'mean_train_accuracy': np.mean([r['train_accuracy'] for r in fold_results])
        }

    def save_results(self, filepath: str) -> None:
        """Save training results to file."""
        results_dict = {
            'train_loss_history': self.result.train_loss_history,
            'val_loss_history': self.result.val_loss_history,
            'train_acc_history': self.result.train_acc_history,
            'val_acc_history': self.result.val_acc_history,
            'best_val_loss': self.result.best_val_loss,
            'best_val_acc': self.result.best_val_acc,
            'total_time': self.result.total_time,
            'n_gates': self.result.n_gates,
            'n_parameters': self.result.n_parameters,
            'total_measurements': self.result.total_measurements
        }

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def load_results(self, filepath: str) -> TrainingResult:
        """Load training results from file."""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)

        self.result = TrainingResult(**results_dict)
        return self.result


class ComparisonTrainer:
    """
    Trainer for comparing adaptive QNN with traditional approaches.

    This trainer facilitates fair comparison between:
    1. Adaptive QNN with analytic estimation
    2. Traditional VQC with gradient descent
    3. Traditional VQC with gradient-free optimization
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int = 2,
        random_state: Optional[int] = None
    ):
        """
        Initialize the comparison trainer.

        Args:
            n_qubits: Number of qubits
            n_classes: Number of classes
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.random_state = random_state
        self.results: Dict[str, Dict] = {}

    def run_comparison(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Run comparison experiments.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            methods: List of methods to compare

        Returns:
            Dictionary with results for each method
        """
        if methods is None:
            methods = ['adaptive', 'standard_cobyla', 'standard_spsa']

        for method in methods:
            print(f"\n{'='*50}")
            print(f"Training: {method}")
            print('='*50)

            start_time = time.time()

            if method == 'adaptive':
                result = self._train_adaptive(X_train, y_train, X_test, y_test)
            elif method == 'standard_cobyla':
                result = self._train_standard(X_train, y_train, X_test, y_test, 'cobyla')
            elif method == 'standard_spsa':
                result = self._train_standard(X_train, y_train, X_test, y_test, 'spsa')
            else:
                continue

            result['training_time'] = time.time() - start_time
            self.results[method] = result

            print(f"Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"Training Time: {result['training_time']:.2f}s")

        return self.results

    def _train_adaptive(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Train using adaptive QNN."""
        from ..models.adaptive_qnn import create_adaptive_qnn

        model = create_adaptive_qnn(
            n_qubits=self.n_qubits,
            n_classes=self.n_classes
        )

        model.fit(X_train, y_train, verbose=True)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_gates': model.get_circuit_info()['n_gates'],
            'n_params': model.get_circuit_info()['n_parameters'],
            'measurements': model.estimator.get_measurement_count()
        }

    def _train_standard(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        optimizer: str
    ) -> Dict:
        """Train using standard VQC with specified optimizer."""
        from ..models.adaptive_qnn import AdaptiveQNN
        from ..estimators.gradient_free import GradientFreeOptimizer

        # Create model with fixed circuit
        model = AdaptiveQNN(
            n_qubits=self.n_qubits,
            n_classes=self.n_classes,
            use_iterative_reconstruction=False
        )

        # Build circuit with fixed layers
        model.build_initial_circuit(X_train.shape[1])
        for _ in range(3):  # 3 variational layers
            model.circuit_builder.add_variational_layer()
        model.circuit = model.circuit_builder.get_circuit()

        # Train with gradient-free optimizer
        gf_optimizer = GradientFreeOptimizer(method=optimizer)

        cost_fn = model._create_cost_function(X_train, y_train)
        optimal_params, history = gf_optimizer.optimize(
            model.circuit,
            model.circuit_builder.get_parameters(),
            cost_fn
        )

        model.trained_params = optimal_params
        model.is_trained = True

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_gates': model.circuit.depth(),
            'n_params': len(optimal_params),
            'measurements': gf_optimizer.get_evaluation_count() * model.shots
        }

    def generate_report(self) -> str:
        """Generate comparison report."""
        report = ["# QNN Training Comparison Report\n"]
        report.append("| Method | Train Acc | Test Acc | Gates | Params | Measurements | Time (s) |")
        report.append("|--------|-----------|----------|-------|--------|--------------|----------|")

        for method, result in self.results.items():
            report.append(
                f"| {method} | {result['train_accuracy']:.4f} | "
                f"{result['test_accuracy']:.4f} | {result['n_gates']} | "
                f"{result['n_params']} | {result['measurements']} | "
                f"{result['training_time']:.2f} |"
            )

        return "\n".join(report)
