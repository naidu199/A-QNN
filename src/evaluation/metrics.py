"""
Evaluation Metrics for Quantum Neural Networks
===============================================

Comprehensive metrics for evaluating QNN performance, including
quantum-specific metrics for barren plateau analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report as sklearn_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Optional prediction probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # Add AUC if probabilities provided
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass  # AUC not computable

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Generate a detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Formatted report string
    """
    return sklearn_report(y_true, y_pred, target_names=class_names)


def barren_plateau_metrics(
    cost_history: List[float],
    gradient_history: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute metrics related to barren plateau detection.

    Barren plateaus are characterized by:
    1. Exponentially vanishing gradients
    2. Cost function concentrated around mean
    3. Flat optimization landscape

    Args:
        cost_history: History of cost values during training
        gradient_history: Optional history of gradient norms

    Returns:
        Dictionary with barren plateau metrics
    """
    costs = np.array(cost_history)

    metrics = {
        'cost_variance': np.var(costs),
        'cost_range': np.max(costs) - np.min(costs),
        'cost_mean': np.mean(costs),
        'cost_std': np.std(costs),
        'relative_variance': np.var(costs) / (np.mean(costs) ** 2 + 1e-10)
    }

    # Detect if training made progress
    if len(costs) > 1:
        metrics['total_improvement'] = costs[0] - costs[-1]
        metrics['improvement_rate'] = metrics['total_improvement'] / len(costs)

        # Check for plateau (no improvement over recent iterations)
        window = min(10, len(costs) // 2)
        if window > 0:
            recent = costs[-window:]
            metrics['recent_variance'] = np.var(recent)
            metrics['is_plateau'] = metrics['recent_variance'] < 1e-6
        else:
            metrics['is_plateau'] = False

    # Gradient-based metrics
    if gradient_history is not None and len(gradient_history) > 0:
        grads = np.array(gradient_history)
        metrics['mean_gradient'] = np.mean(grads)
        metrics['max_gradient'] = np.max(grads)
        metrics['gradient_variance'] = np.var(grads)

        # Check for vanishing gradients
        metrics['vanishing_gradients'] = metrics['mean_gradient'] < 1e-5

    # Overall barren plateau assessment
    is_barren = (
        metrics['cost_variance'] < 0.01 and
        metrics.get('is_plateau', False)
    )
    metrics['barren_plateau_detected'] = is_barren

    return metrics


def circuit_efficiency_score(
    n_gates: int,
    n_parameters: int,
    accuracy: float,
    circuit_depth: int,
    n_qubits: int
) -> Dict[str, float]:
    """
    Compute circuit efficiency metrics.

    Measures how efficiently the circuit achieves its performance.

    Args:
        n_gates: Number of gates in circuit
        n_parameters: Number of trainable parameters
        accuracy: Model accuracy
        circuit_depth: Circuit depth
        n_qubits: Number of qubits

    Returns:
        Dictionary with efficiency metrics
    """
    # Gates per accuracy point
    if accuracy > 0:
        gates_per_accuracy = n_gates / accuracy
    else:
        gates_per_accuracy = float('inf')

    # Parameters per accuracy point
    if accuracy > 0:
        params_per_accuracy = n_parameters / accuracy
    else:
        params_per_accuracy = float('inf')

    # Depth efficiency (accuracy per depth)
    if circuit_depth > 0:
        depth_efficiency = accuracy / circuit_depth
    else:
        depth_efficiency = 0

    # Overall efficiency score (higher is better)
    # Balances accuracy against resource usage
    if n_gates > 0 and circuit_depth > 0:
        efficiency = accuracy ** 2 / (n_gates * np.log1p(circuit_depth))
    else:
        efficiency = 0

    # Expressibility proxy (parameters relative to circuit capacity)
    circuit_capacity = n_qubits * circuit_depth
    if circuit_capacity > 0:
        expressibility_ratio = n_parameters / circuit_capacity
    else:
        expressibility_ratio = 0

    return {
        'gates_per_accuracy': gates_per_accuracy,
        'params_per_accuracy': params_per_accuracy,
        'depth_efficiency': depth_efficiency,
        'overall_efficiency': efficiency,
        'expressibility_ratio': expressibility_ratio,
        'gate_density': n_gates / max(circuit_depth * n_qubits, 1)
    }


def compare_models(
    results: List[Dict[str, Any]],
    model_names: List[str]
) -> Dict[str, Any]:
    """
    Compare multiple model results.

    Args:
        results: List of result dictionaries
        model_names: Names of models

    Returns:
        Comparison summary
    """
    comparison = {
        'models': model_names,
        'metrics': {}
    }

    # Extract common metrics
    metric_keys = ['accuracy', 'f1_score', 'n_gates', 'n_parameters',
                   'training_time', 'total_measurements']

    for key in metric_keys:
        values = []
        for result in results:
            if key in result:
                values.append(result[key])
            else:
                values.append(None)
        comparison['metrics'][key] = values

    # Determine best model for each metric
    comparison['best'] = {}
    for key, values in comparison['metrics'].items():
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
        if valid_values:
            if key in ['accuracy', 'f1_score']:
                best_idx = max(valid_values, key=lambda x: x[1])[0]
            else:  # Lower is better for resources
                best_idx = min(valid_values, key=lambda x: x[1])[0]
            comparison['best'][key] = model_names[best_idx]

    return comparison


def quantum_advantage_score(
    qnn_accuracy: float,
    classical_accuracy: float,
    qnn_training_time: float,
    classical_training_time: float,
    n_qubits: int,
    n_samples: int
) -> Dict[str, float]:
    """
    Estimate quantum advantage metrics.

    Args:
        qnn_accuracy: QNN accuracy
        classical_accuracy: Classical model accuracy
        qnn_training_time: QNN training time
        classical_training_time: Classical training time
        n_qubits: Number of qubits
        n_samples: Training sample size

    Returns:
        Quantum advantage metrics
    """
    # Accuracy advantage
    accuracy_diff = qnn_accuracy - classical_accuracy
    accuracy_ratio = qnn_accuracy / max(classical_accuracy, 0.01)

    # Time comparison
    time_ratio = classical_training_time / max(qnn_training_time, 0.01)

    # Estimated scaling advantage
    # QNN: O(poly(n_qubits) * n_samples)
    # Classical: O(n_features^2 * n_samples) for SVM-like
    classical_scaling = (2 ** n_qubits) ** 2 * n_samples
    quantum_scaling = n_qubits ** 2 * n_samples
    scaling_advantage = classical_scaling / max(quantum_scaling, 1)

    return {
        'accuracy_advantage': accuracy_diff,
        'accuracy_ratio': accuracy_ratio,
        'time_ratio': time_ratio,
        'theoretical_scaling_advantage': np.log2(scaling_advantage),
        'overall_advantage': accuracy_ratio * time_ratio
    }
