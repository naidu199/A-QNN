"""
Dataset Comparison Example
===========================

This example compares the Adaptive QNN performance across multiple datasets.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import AdaptiveQNN
from src.data import (
    load_moons_quantum,
    load_circles_quantum,
    load_iris_quantum,
    generate_quantum_data
)
from src.evaluation import compute_metrics


def run_experiment(dataset_name, X_train, X_test, y_train, y_test, n_qubits):
    """Run a single experiment and return results."""

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")

    # Create and train model
    model = AdaptiveQNN(
        n_qubits=n_qubits,
        n_classes=len(np.unique(y_train)),
        max_gates=20,
        shots=512
    )

    model.fit(X_train, y_train, max_iterations=10, verbose=False)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    circuit_info = model.get_circuit_info()

    results = {
        'dataset': dataset_name,
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'f1_score': test_metrics['f1_score'],
        'n_gates': circuit_info['n_gates'],
        'n_params': circuit_info['n_parameters'],
        'depth': circuit_info['depth']
    }

    print(f"\nResults:")
    print(f"  Train Accuracy: {results['train_acc']:.4f}")
    print(f"  Test Accuracy:  {results['test_acc']:.4f}")
    print(f"  F1 Score:       {results['f1_score']:.4f}")
    print(f"  Circuit Depth:  {results['depth']}")
    print(f"  Gates:          {results['n_gates']}")

    return results


def main():
    """Run comparison across datasets."""

    print("\n" + "="*60)
    print("Adaptive QNN - Dataset Comparison")
    print("="*60)

    np.random.seed(42)
    n_qubits = 4
    all_results = []

    # 1. Moons dataset
    X_train, X_test, y_train, y_test = load_moons_quantum(
        n_samples=150, n_qubits=n_qubits, noise=0.15, random_state=42
    )
    results = run_experiment("Moons", X_train, X_test, y_train, y_test, n_qubits)
    all_results.append(results)

    # 2. Circles dataset
    X_train, X_test, y_train, y_test = load_circles_quantum(
        n_samples=150, n_qubits=n_qubits, noise=0.1, random_state=42
    )
    results = run_experiment("Circles", X_train, X_test, y_train, y_test, n_qubits)
    all_results.append(results)

    # 3. Iris dataset
    X_train, X_test, y_train, y_test = load_iris_quantum(
        n_qubits=n_qubits, binary=True, random_state=42
    )
    results = run_experiment("Iris (binary)", X_train, X_test, y_train, y_test, n_qubits)
    all_results.append(results)

    # 4. XOR pattern
    X_train, X_test, y_train, y_test = generate_quantum_data(
        n_samples=150, n_features=n_qubits, pattern='xor', noise=0.1, random_state=42
    )
    results = run_experiment("XOR Pattern", X_train, X_test, y_train, y_test, n_qubits)
    all_results.append(results)

    # Summary table
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"{'Dataset':<15} {'Train Acc':>10} {'Test Acc':>10} {'F1':>10} {'Gates':>8} {'Depth':>8}")
    print("-"*70)

    for r in all_results:
        print(f"{r['dataset']:<15} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} "
              f"{r['f1_score']:>10.4f} {r['n_gates']:>8} {r['depth']:>8}")

    print("-"*70)

    # Averages
    avg_train = np.mean([r['train_acc'] for r in all_results])
    avg_test = np.mean([r['test_acc'] for r in all_results])
    avg_f1 = np.mean([r['f1_score'] for r in all_results])
    avg_gates = np.mean([r['n_gates'] for r in all_results])

    print(f"{'Average':<15} {avg_train:>10.4f} {avg_test:>10.4f} {avg_f1:>10.4f} {avg_gates:>8.1f}")

    print("\n" + "="*70)
    print("Comparison Complete!")
    print("="*70 + "\n")

    return all_results


if __name__ == '__main__':
    results = main()
