"""
QNN Classifier - Scikit-learn Compatible Wrapper
=================================================

This module provides a scikit-learn compatible wrapper for the Adaptive QNN,
enabling easy integration with standard ML pipelines and tools.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .adaptive_qnn import AdaptiveQNN


class QNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum Neural Network Classifier with scikit-learn interface.

    This class wraps the AdaptiveQNN to provide a familiar sklearn-style API,
    making it easy to use in standard machine learning workflows including:
    - Cross-validation
    - Grid search for hyperparameter tuning
    - Pipeline integration
    - Standard metrics evaluation

    Example:
        >>> from src.models import QNNClassifier
        >>> from sklearn.model_selection import cross_val_score
        >>>
        >>> clf = QNNClassifier(n_qubits=4, max_gates=30)
        >>> scores = cross_val_score(clf, X, y, cv=5)
        >>> print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

    Attributes:
        n_qubits: Number of qubits in the quantum circuit
        n_classes: Number of output classes (set during fit)
        model_: The trained AdaptiveQNN model
        classes_: Unique class labels
        label_encoder_: Encoder for class labels
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding_type: str = 'angle',
        max_gates: int = 30,
        shots: int = 1024,
        measurement_budget: int = 50000,
        max_iterations: int = 10,
        improvement_threshold: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the QNN Classifier.

        Args:
            n_qubits: Number of qubits in the circuit
            encoding_type: Data encoding strategy
            max_gates: Maximum gates in adaptive construction
            shots: Measurement shots per evaluation
            measurement_budget: Total measurement budget
            max_iterations: Maximum training iterations
            improvement_threshold: Convergence threshold
            random_state: Random seed for reproducibility
            verbose: Print training progress
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.max_gates = max_gates
        self.shots = shots
        self.measurement_budget = measurement_budget
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QNNClassifier':
        """
        Fit the quantum neural network classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            self
        """
        # Validate input
        X, y = check_X_y(X, y)

        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        # Store classes
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        # Create and train model
        self.model_ = AdaptiveQNN(
            n_qubits=self.n_qubits,
            n_classes=self.n_classes_,
            encoding_type=self.encoding_type,
            max_gates=self.max_gates,
            shots=self.shots,
            measurement_budget=self.measurement_budget
        )

        self.model_.fit(
            X, y_encoded,
            max_iterations=self.max_iterations,
            improvement_threshold=self.improvement_threshold,
            verbose=self.verbose
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        check_is_fitted(self, ['model_', 'label_encoder_'])
        X = check_array(X)

        y_pred = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        check_is_fitted(self, ['model_'])
        X = check_array(X)

        return self.model_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'n_qubits': self.n_qubits,
            'encoding_type': self.encoding_type,
            'max_gates': self.max_gates,
            'shots': self.shots,
            'measurement_budget': self.measurement_budget,
            'max_iterations': self.max_iterations,
            'improvement_threshold': self.improvement_threshold,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> 'QNNClassifier':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the trained circuit."""
        check_is_fitted(self, ['model_'])
        return self.model_.get_circuit_info()

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        check_is_fitted(self, ['model_'])
        return self.model_.get_training_history()


class QNNRegressor(BaseEstimator):
    """
    Quantum Neural Network Regressor with scikit-learn interface.

    Similar to QNNClassifier but for regression tasks.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding_type: str = 'angle',
        max_gates: int = 30,
        shots: int = 1024,
        measurement_budget: int = 50000,
        max_iterations: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize the QNN Regressor."""
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.max_gates = max_gates
        self.shots = shots
        self.measurement_budget = measurement_budget
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.verbose = verbose

        # Regression-specific attributes
        self.y_mean_ = None
        self.y_std_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QNNRegressor':
        """Fit the quantum neural network regressor."""
        X, y = check_X_y(X, y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Normalize targets
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y) if np.std(y) > 0 else 1.0
        y_normalized = (y - self.y_mean_) / self.y_std_

        # Create model (n_classes=1 for regression output)
        self.model_ = AdaptiveQNN(
            n_qubits=self.n_qubits,
            n_classes=1,
            encoding_type=self.encoding_type,
            max_gates=self.max_gates,
            shots=self.shots,
            measurement_budget=self.measurement_budget
        )

        # Custom training for regression
        self._fit_regression(X, y_normalized)

        return self

    def _fit_regression(self, X: np.ndarray, y: np.ndarray) -> None:
        """Internal regression training."""
        # Build initial circuit
        self.model_.build_initial_circuit(X.shape[1])

        # Create regression cost function
        def cost_fn(circuit, params):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                pred = self.model_._forward_single(circuit, params, xi)
                loss = (pred[0] - yi) ** 2
                total_loss += loss
            return total_loss / len(X)

        # Use the estimator for training
        self.model_.circuit, self.model_.trained_params, history = \
            self.model_.estimator.reconstruct_circuit(
                self.model_.circuit,
                self.model_.gate_pool,
                cost_fn,
                max_gates=self.max_gates,
                improvement_threshold=1e-4
            )

        self.model_.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        check_is_fitted(self, ['model_', 'y_mean_', 'y_std_'])
        X = check_array(X)

        X_processed = self.model_.data_encoder.preprocess_data(X, method='minmax')
        predictions = []

        for xi in X_processed:
            exp_val = self.model_._forward_single(
                self.model_.circuit,
                self.model_.trained_params,
                xi
            )
            pred = exp_val[0] * self.y_std_ + self.y_mean_
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
