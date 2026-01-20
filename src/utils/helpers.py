"""
Helper Utilities
=================

General utility functions for the A-QNN project.
"""

import numpy as np
import time
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
from contextlib import contextmanager


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'numpy_version': np.__version__,
    }

    # Check for Qiskit
    try:
        import qiskit
        info['qiskit_version'] = qiskit.__version__

        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        info['aer_available'] = True
        info['max_qubits'] = backend.configuration().n_qubits
    except ImportError:
        info['qiskit_available'] = False

    # Check for PyTorch
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['torch_available'] = False

    return info


def save_model(
    model: Any,
    filepath: str,
    include_circuit: bool = True
) -> None:
    """
    Save a trained QNN model.

    Args:
        model: Trained model to save
        filepath: Path to save to
        include_circuit: Whether to include circuit diagram
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'trained_params': {str(k): v for k, v in model.trained_params.items()},
        'n_qubits': model.n_qubits,
        'n_classes': model.n_classes,
        'encoding_type': model.encoding_type,
        'max_gates': model.max_gates,
        'training_history': model.training_history,
        'is_trained': model.is_trained
    }

    # Save circuit as QASM
    if include_circuit and model.circuit is not None:
        try:
            save_dict['circuit_qasm'] = model.circuit.qasm()
        except:
            pass

    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)


def load_model(
    filepath: str,
    model_class: Optional[type] = None
) -> Any:
    """
    Load a saved QNN model.

    Args:
        filepath: Path to saved model
        model_class: Optional model class to instantiate

    Returns:
        Loaded model or dictionary
    """
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)

    if model_class is not None:
        model = model_class(
            n_qubits=save_dict['n_qubits'],
            n_classes=save_dict['n_classes'],
            encoding_type=save_dict['encoding_type'],
            max_gates=save_dict['max_gates']
        )

        # Restore circuit from QASM if available
        if 'circuit_qasm' in save_dict:
            from qiskit import QuantumCircuit
            model.circuit = QuantumCircuit.from_qasm_str(save_dict['circuit_qasm'])

        model.training_history = save_dict['training_history']
        model.is_trained = save_dict['is_trained']

        return model

    return save_dict


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing operations.

    Example:
        >>> with timer("Training"):
        ...     model.fit(X, y)
        Training completed in 5.23 seconds
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} completed in {elapsed:.2f} seconds")


def timing_decorator(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} executed in {elapsed:.2f} seconds")
        return result
    return wrapper


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_model_summary(model: Any) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: QNN model
    """
    print("\n" + "="*50)
    print("Adaptive QNN Model Summary")
    print("="*50)

    if hasattr(model, 'get_circuit_info'):
        info = model.get_circuit_info()
        print(f"Qubits:          {info.get('n_qubits', 'N/A')}")
        print(f"Circuit Depth:   {info.get('depth', 'N/A')}")
        print(f"Parameters:      {info.get('n_parameters', 'N/A')}")
        print(f"Total Gates:     {info.get('n_gates', 'N/A')}")
        print(f"Encoding:        {info.get('encoding_type', 'N/A')}")

        if 'gate_counts' in info:
            print("\nGate Breakdown:")
            for gate, count in info['gate_counts'].items():
                print(f"  {gate}: {count}")

    print("="*50 + "\n")


def estimate_resource_requirements(
    n_qubits: int,
    max_gates: int,
    shots: int,
    n_samples: int
) -> Dict[str, Any]:
    """
    Estimate computational resources needed for training.

    Args:
        n_qubits: Number of qubits
        max_gates: Maximum gates
        shots: Shots per measurement
        n_samples: Training samples

    Returns:
        Resource estimates
    """
    # State vector size
    state_size = 2 ** n_qubits * 16  # Complex128

    # Estimated circuit evaluations
    # For adaptive: 3 measurements per parameter for analytic estimation
    max_parameters = max_gates  # Rough estimate
    circuit_evals = max_parameters * 3 * max_gates

    # Total shots
    total_shots = circuit_evals * shots * n_samples

    # Estimated memory (rough)
    memory_mb = state_size / (1024 ** 2)

    # Estimated time (very rough: 1ms per circuit on simulator)
    estimated_time = circuit_evals * n_samples * 0.001

    return {
        'state_vector_size_mb': memory_mb,
        'estimated_circuit_evaluations': circuit_evals,
        'total_shots': total_shots,
        'estimated_time_seconds': estimated_time,
        'estimated_time_formatted': format_time(estimated_time)
    }
