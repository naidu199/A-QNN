"""
Evaluation and metrics module.
"""

from .metrics import (
    compute_metrics,
    classification_report,
    barren_plateau_metrics,
    circuit_efficiency_score
)
from .visualization import (
    plot_training_history,
    plot_circuit,
    plot_decision_boundary,
    plot_barren_plateau_analysis
)
from .ibm_runner import IBMQuantumRunner
from .comparison import QNNComparisonPipeline

__all__ = [
    "compute_metrics",
    "classification_report",
    "barren_plateau_metrics",
    "circuit_efficiency_score",
    "plot_training_history",
    "plot_circuit",
    "plot_decision_boundary",
    "plot_barren_plateau_analysis",
    "IBMQuantumRunner",
    "QNNComparisonPipeline",
]
