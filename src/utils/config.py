"""
Configuration Management
=========================

Configuration classes and utilities for A-QNN experiments.
"""

import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class QNNConfig:
    """
    Configuration for Adaptive QNN experiments.

    This dataclass holds all hyperparameters and settings needed
    for training and evaluating adaptive QNNs.
    """

    # Model architecture
    n_qubits: int = 4
    n_classes: int = 2
    encoding_type: str = 'angle'
    max_gates: int = 30

    # Training parameters
    max_iterations: int = 20
    improvement_threshold: float = 1e-4
    shots: int = 1024
    measurement_budget: int = 50000

    # Data parameters
    test_size: float = 0.2
    validation_split: float = 0.1
    batch_size: Optional[int] = None

    # Optimization
    use_iterative_reconstruction: bool = True
    estimator_type: str = 'analytic'  # 'analytic', 'fourier', 'gradient_free'

    # Regularization
    gate_penalty: float = 0.0  # Penalty for number of gates
    depth_penalty: float = 0.0  # Penalty for circuit depth

    # Logging and checkpointing
    verbose: bool = True
    log_interval: int = 1
    checkpoint_dir: Optional[str] = None

    # Random state
    random_state: Optional[int] = 42

    # Hardware settings
    backend: str = 'aer_simulator'
    noise_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QNNConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items()
                     if k in cls.__dataclass_fields__})

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        path = Path(filepath)
        config_dict = self.to_dict()

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'QNNConfig':
        """Load configuration from file."""
        path = Path(filepath)

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)

        return cls.from_dict(config_dict)


@dataclass
class ExperimentConfig:
    """
    Configuration for a complete experiment.

    Includes model config plus experiment-specific settings.
    """

    # Experiment metadata
    name: str = "adaptive_qnn_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Dataset
    dataset: str = 'moons'  # 'moons', 'circles', 'iris', 'custom'
    dataset_params: Dict[str, Any] = field(default_factory=dict)

    # Model configuration
    model_config: QNNConfig = field(default_factory=QNNConfig)

    # Experiment settings
    n_runs: int = 1  # Number of repeated runs
    cross_validation_folds: int = 0  # 0 = no CV

    # Comparison baselines
    run_baselines: bool = False
    baseline_methods: List[str] = field(
        default_factory=lambda: ['standard_cobyla', 'standard_spsa']
    )

    # Output settings
    output_dir: str = 'results'
    save_models: bool = True
    save_plots: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['model_config'] = self.model_config.to_dict()
        return d

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        if 'model_config' in config_dict:
            config_dict['model_config'] = QNNConfig.from_dict(
                config_dict['model_config']
            )
        return cls(**{k: v for k, v in config_dict.items()
                     if k in cls.__dataclass_fields__})


def load_config(filepath: str) -> QNNConfig:
    """
    Load configuration from file.

    Args:
        filepath: Path to config file (YAML or JSON)

    Returns:
        QNNConfig instance
    """
    return QNNConfig.load(filepath)


def save_config(config: QNNConfig, filepath: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        filepath: Destination path
    """
    config.save(filepath)


# Preset configurations
PRESETS = {
    'small': QNNConfig(
        n_qubits=2,
        max_gates=10,
        max_iterations=10,
        shots=512
    ),
    'medium': QNNConfig(
        n_qubits=4,
        max_gates=30,
        max_iterations=20,
        shots=1024
    ),
    'large': QNNConfig(
        n_qubits=8,
        max_gates=50,
        max_iterations=30,
        shots=2048
    ),
    'quick_test': QNNConfig(
        n_qubits=2,
        max_gates=5,
        max_iterations=5,
        shots=256,
        measurement_budget=5000
    )
}


def get_preset(name: str) -> QNNConfig:
    """
    Get a preset configuration.

    Args:
        name: Preset name ('small', 'medium', 'large', 'quick_test')

    Returns:
        QNNConfig with preset values
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
