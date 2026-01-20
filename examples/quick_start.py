"""
Quick Start Example: Adaptive Quantum Neural Network
=====================================================

This example demonstrates the basic usage of the Adaptive QNN
for a simple classification task.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import AdaptiveQNN
from src.data import load_moons_quantum
from src.evaluation import compute_metrics


def main():
    """Run a quick example of Adaptive QNN training."""

    print("="*60)
    print("Adaptive QNN Quick Start Example")
    print("="*60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Load data
    print("\n1. Loading make_moons dataset...")
    X_train, X_test, y_train, y_test = load_moons_quantum(
        n_samples=100,
        n_qubits=2,
        noise=0.1,
        random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # 2. Create model
    print("\n2. Creating Adaptive QNN model...")
    model = AdaptiveQNN(
        n_qubits=2,
        n_classes=2,
        encoding_type='angle',
        max_gates=15,
        shots=512
    )
    print("   Model created successfully!")

    # 3. Train model
    print("\n3. Training model...")
    print("-"*40)
    model.fit(
        X_train, y_train,
        max_iterations=10,
        improvement_threshold=1e-3,
        verbose=True
    )

    # 4. Evaluate
    print("\n4. Evaluating model...")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)

    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy:     {test_acc:.4f}")

    # 5. Circuit info
    print("\n5. Circuit Information:")
    info = model.get_circuit_info()
    print(f"   Qubits:      {info['n_qubits']}")
    print(f"   Depth:       {info['depth']}")
    print(f"   Gates:       {info['n_gates']}")
    print(f"   Parameters:  {info['n_parameters']}")

    # 6. Show circuit (text representation)
    print("\n6. Quantum Circuit:")
    print(model.circuit.draw(output='text'))

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)

    return model


if __name__ == '__main__':
    model = main()
