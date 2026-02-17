"""
Adaptive Quantum Neural Network (A-QNN)
========================================

A novel approach to training Quantum Neural Networks that addresses the barren plateau
problem through adaptive circuit construction and analytic iterative reconstruction.

Modules:
--------
- circuits: Quantum circuit construction and gate operations
- estimators: Analytic parameter estimation methods
- models: Adaptive QNN model implementations
- training: Training utilities and optimization
- data: Data preprocessing and encoding
- evaluation: Metrics and evaluation tools
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "A-QNN Research Team"

from .models.adaptive_qnn import AdaptiveQNN
from .models.fixed_ansatz_qnn import FixedAnsatzQNN
from .training.trainer import QNNTrainer
from .estimators.arc_estimator import ARCEstimator
from .evaluation.ibm_runner import IBMQuantumRunner
from .evaluation.comparison import QNNComparisonPipeline

__all__ = [
    "AdaptiveQNN",
    "FixedAnsatzQNN",
    "QNNTrainer",
    "ARCEstimator",
    "IBMQuantumRunner",
    "QNNComparisonPipeline",
]
