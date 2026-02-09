"""
Hyperparameter Tuning Utilities
================================

This module provides utilities for tuning Adaptive QNN hyperparameters
to optimize accuracy and training efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from itertools import product
import json
from pathlib import Path
from datetime import datetime

from ..models import AdaptiveQNN


class HyperparameterTuner:
    """
    Hyperparameter tuner for Adaptive QNN models.

    Supports grid search and random search over hyperparameter spaces.
    """

    # Default hyperparameter search space
    DEFAULT_SEARCH_SPACE = {
        'max_gates': [10, 15, 20, 25],
        'max_iterations': [8, 12, 16],
        'beam_width': [1, 2, 3],
        'batch_size': [20, 32, 48],
        'improvement_threshold': [1e-4, 1e-5, 1e-6]
    }

    def __init__(
        self,
        n_qubits: int,
        n_classes: int = 2,
        search_space: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 3,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the hyperparameter tuner.

        Args:
            n_qubits: Number of qubits for the QNN
            n_classes: Number of output classes
            search_space: Dictionary of hyperparameters and their possible values
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            verbose: Print progress during tuning
        """
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

        self.results_: List[Dict] = []
        self.best_params_: Optional[Dict] = None
        self.best_score_: float = 0.0

    def grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform grid search over all hyperparameter combinations.

        Args:
            X: Training features
            y: Training labels
            scoring: Scoring metric ('accuracy', 'f1', 'precision', 'recall')

        Returns:
            Dict with best parameters and all results
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Generate all combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        all_combinations = list(product(*param_values))

        if self.verbose:
            print(f"Grid Search: {len(all_combinations)} configurations to test")
            print(f"CV folds: {self.cv_folds}")

        self.results_ = []
        self.best_score_ = 0.0
        self.best_params_ = None

        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))

            if self.verbose:
                print(f"\n[{i+1}/{len(all_combinations)}] Testing: {params}")

            try:
                score = self._cross_validate(X, y, params, scoring)

                self.results_.append({
                    'params': params,
                    'score': score,
                    'status': 'success'
                })

                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params.copy()
                    if self.verbose:
                        print(f"  New best score: {score:.4f}")
                else:
                    if self.verbose:
                        print(f"  Score: {score:.4f}")

            except Exception as e:
                self.results_.append({
                    'params': params,
                    'score': 0.0,
                    'status': 'failed',
                    'error': str(e)
                })
                if self.verbose:
                    print(f"  Failed: {e}")

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'all_results': self.results_
        }

    def random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 20,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform random search over hyperparameter space.

        Args:
            X: Training features
            y: Training labels
            n_iter: Number of random configurations to test
            scoring: Scoring metric

        Returns:
            Dict with best parameters and all results
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.verbose:
            print(f"Random Search: {n_iter} configurations to test")
            print(f"CV folds: {self.cv_folds}")

        self.results_ = []
        self.best_score_ = 0.0
        self.best_params_ = None

        for i in range(n_iter):
            # Sample random parameters
            params = {
                name: np.random.choice(values)
                for name, values in self.search_space.items()
            }

            if self.verbose:
                print(f"\n[{i+1}/{n_iter}] Testing: {params}")

            try:
                score = self._cross_validate(X, y, params, scoring)

                self.results_.append({
                    'params': params,
                    'score': score,
                    'status': 'success'
                })

                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params.copy()
                    if self.verbose:
                        print(f"  New best score: {score:.4f}")
                else:
                    if self.verbose:
                        print(f"  Score: {score:.4f}")

            except Exception as e:
                self.results_.append({
                    'params': params,
                    'score': 0.0,
                    'status': 'failed',
                    'error': str(e)
                })
                if self.verbose:
                    print(f"  Failed: {e}")

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'all_results': self.results_
        }

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        scoring: str
    ) -> float:
        """
        Perform k-fold cross-validation with given parameters.
        """
        n_samples = len(X)
        fold_size = n_samples // self.cv_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        scores = []

        for fold in range(self.cv_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.cv_folds - 1 else n_samples

            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model with current parameters
            model = AdaptiveQNN(
                n_qubits=self.n_qubits,
                n_classes=self.n_classes,
                max_gates=params.get('max_gates', 15)
            )

            model.fit(
                X_train, y_train,
                max_iterations=params.get('max_iterations', 10),
                beam_width=params.get('beam_width', 1),
                batch_size=params.get('batch_size', 32),
                improvement_threshold=params.get('improvement_threshold', 1e-4),
                verbose=False
            )

            # Evaluate
            fold_score = self._compute_score(model, X_val, y_val, scoring)
            scores.append(fold_score)

        return np.mean(scores)

    def _compute_score(
        self,
        model: AdaptiveQNN,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str
    ) -> float:
        """Compute the specified scoring metric."""
        predictions = model.predict(X)

        if scoring == 'accuracy':
            return np.mean(predictions == y)
        elif scoring == 'f1':
            from sklearn.metrics import f1_score
            return f1_score(y, predictions, average='weighted')
        elif scoring == 'precision':
            from sklearn.metrics import precision_score
            return precision_score(y, predictions, average='weighted', zero_division=0)
        elif scoring == 'recall':
            from sklearn.metrics import recall_score
            return recall_score(y, predictions, average='weighted', zero_division=0)
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")

    def save_results(self, filepath: str) -> None:
        """Save tuning results to a JSON file."""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'n_qubits': self.n_qubits,
            'n_classes': self.n_classes,
            'search_space': {k: [float(v) if isinstance(v, (int, float)) else v
                                 for v in vals] for k, vals in self.search_space.items()},
            'cv_folds': self.cv_folds,
            'best_params': self.best_params_,
            'best_score': float(self.best_score_),
            'all_results': [
                {
                    'params': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                              for k, v in r['params'].items()},
                    'score': float(r['score']),
                    'status': r['status']
                }
                for r in self.results_
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        if self.verbose:
            print(f"Results saved to {filepath}")


def quick_tune(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    n_iter: int = 10,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Quick hyperparameter tuning with sensible defaults.

    Args:
        X: Training features
        y: Training labels
        n_qubits: Number of qubits
        n_iter: Number of random configurations to test
        random_state: Random seed

    Returns:
        Tuple of (best_params, best_score)
    """
    # Focused search space for quick tuning
    search_space = {
        'max_gates': [12, 18, 24],
        'max_iterations': [10, 15],
        'beam_width': [1, 2, 3],
        'batch_size': [24, 40],
        'improvement_threshold': [1e-5, 1e-6]
    }

    tuner = HyperparameterTuner(
        n_qubits=n_qubits,
        n_classes=len(np.unique(y)),
        search_space=search_space,
        cv_folds=3,
        random_state=random_state,
        verbose=True
    )

    results = tuner.random_search(X, y, n_iter=n_iter)
    return results['best_params'], results['best_score']


def analyze_beam_width_impact(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    beam_widths: List[int] = [1, 2, 3, 4, 5],
    n_trials: int = 3
) -> Dict[str, Any]:
    """
    Analyze the impact of beam width on model performance.

    Args:
        X: Training features
        y: Training labels
        n_qubits: Number of qubits
        beam_widths: List of beam widths to test
        n_trials: Number of trials per beam width

    Returns:
        Dict with analysis results
    """
    results = {bw: [] for bw in beam_widths}

    n_samples = len(X)
    n_train = int(0.8 * n_samples)

    for trial in range(n_trials):
        # Shuffle and split
        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for bw in beam_widths:
            model = AdaptiveQNN(n_qubits=n_qubits, n_classes=2, max_gates=20)
            model.fit(
                X_train, y_train,
                max_iterations=12,
                beam_width=bw,
                batch_size=32,
                verbose=False
            )

            score = model.score(X_test, y_test)
            results[bw].append(score)
            print(f"Trial {trial+1}, beam_width={bw}: {score:.4f}")

    # Compute statistics
    summary = {}
    for bw in beam_widths:
        scores = results[bw]
        summary[bw] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }

    return {
        'raw_results': results,
        'summary': summary,
        'best_beam_width': max(summary.keys(), key=lambda k: summary[k]['mean'])
    }
