#!/usr/bin/env python
"""
Main Training Script for Adaptive Quantum Neural Network
==========================================================

This script provides a command-line interface for training and evaluating
the Adaptive QNN on various datasets.

Usage:
    python train.py --dataset moons --n_qubits 4 --max_gates 30
    python train.py --config config.yaml
    python train.py --dataset iris --compare
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import AdaptiveQNN
from src.models.qnn_classifier import QNNClassifier
from src.training import QNNTrainer, TrainingConfig
from src.data import (
    load_iris_quantum, load_moons_quantum,
    load_circles_quantum, generate_quantum_data
)
from src.evaluation import compute_metrics, plot_training_history
from src.utils import set_random_seed, timer, QNNConfig, get_preset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Adaptive Quantum Neural Network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset options
    parser.add_argument('--dataset', type=str, default='moons',
                        choices=['moons', 'circles', 'iris', 'xor', 'parity'],
                        help='Dataset to use')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of samples for synthetic datasets')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Noise level for synthetic datasets')

    # Model options
    parser.add_argument('--n_qubits', type=int, default=4,
                        help='Number of qubits')
    parser.add_argument('--encoding', type=str, default='angle',
                        choices=['angle', 'amplitude', 'iqp'],
                        help='Data encoding type')
    parser.add_argument('--max_gates', type=int, default=30,
                        help='Maximum number of gates')

    # Training options
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='Maximum training iterations')
    parser.add_argument('--shots', type=int, default=1024,
                        help='Measurement shots per evaluation')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='Improvement threshold for convergence')

    # Experiment options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with baseline methods')
    parser.add_argument('--cv', type=int, default=0,
                        help='Number of cross-validation folds (0 for none)')

    # Output options
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save training plots')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['small', 'medium', 'large', 'quick_test'],
                        help='Use preset configuration')

    return parser.parse_args()


def load_dataset(args):
    """Load the specified dataset."""
    if args.dataset == 'moons':
        return load_moons_quantum(
            n_samples=args.n_samples,
            n_qubits=args.n_qubits,
            noise=args.noise,
            random_state=args.seed
        )
    elif args.dataset == 'circles':
        return load_circles_quantum(
            n_samples=args.n_samples,
            n_qubits=args.n_qubits,
            noise=args.noise,
            random_state=args.seed
        )
    elif args.dataset == 'iris':
        return load_iris_quantum(
            n_qubits=args.n_qubits,
            random_state=args.seed
        )
    elif args.dataset in ['xor', 'parity']:
        return generate_quantum_data(
            n_samples=args.n_samples,
            n_features=args.n_qubits,
            pattern=args.dataset,
            noise=args.noise,
            random_state=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    """Main training function."""
    args = parse_args()
    verbose = not args.quiet

    # Set random seed
    set_random_seed(args.seed)

    # Load configuration
    if args.config:
        config = QNNConfig.load(args.config)
    elif args.preset:
        config = get_preset(args.preset)
    else:
        config = QNNConfig(
            n_qubits=args.n_qubits,
            encoding_type=args.encoding,
            max_gates=args.max_gates,
            max_iterations=args.max_iterations,
            shots=args.shots,
            improvement_threshold=args.threshold,
            verbose=verbose,
            random_state=args.seed
        )

    if verbose:
        print("\n" + "="*60)
        print("Adaptive Quantum Neural Network Training")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Dataset:      {args.dataset}")
        print(f"  Qubits:       {config.n_qubits}")
        print(f"  Encoding:     {config.encoding_type}")
        print(f"  Max Gates:    {config.max_gates}")
        print(f"  Max Iter:     {config.max_iterations}")
        print(f"  Shots:        {config.shots}")
        print("")

    # Load data
    if verbose:
        print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset(args)

    if verbose:
        print(f"  Training samples:   {len(X_train)}")
        print(f"  Test samples:       {len(X_test)}")
        print(f"  Features:           {X_train.shape[1]}")
        print(f"  Classes:            {len(np.unique(y_train))}")

    # Create and train model
    if verbose:
        print("\nCreating Adaptive QNN model...")

    model = AdaptiveQNN(
        n_qubits=config.n_qubits,
        n_classes=len(np.unique(y_train)),
        encoding_type=config.encoding_type,
        max_gates=config.max_gates,
        shots=config.shots,
        measurement_budget=config.measurement_budget
    )

    # Train
    if verbose:
        print("\nStarting training...\n")

    with timer("Training"):
        model.fit(
            X_train, y_train,
            max_iterations=config.max_iterations,
            improvement_threshold=config.improvement_threshold,
            verbose=verbose
        )

    # Evaluate
    if verbose:
        print("\nEvaluating model...")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test)

    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"\nTraining Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test Precision:     {test_metrics['precision']:.4f}")
    print(f"Test Recall:        {test_metrics['recall']:.4f}")
    print(f"Test F1 Score:      {test_metrics['f1_score']:.4f}")

    if 'auc_roc' in test_metrics:
        print(f"Test AUC-ROC:       {test_metrics['auc_roc']:.4f}")

    # Circuit info
    circuit_info = model.get_circuit_info()
    print(f"\nCircuit Statistics:")
    print(f"  Depth:        {circuit_info['depth']}")
    print(f"  Gates:        {circuit_info['n_gates']}")
    print(f"  Parameters:   {circuit_info['n_parameters']}")
    print(f"  Measurements: {model.estimator.get_measurement_count()}")

    # Cross-validation
    if args.cv > 0:
        print(f"\nRunning {args.cv}-fold cross-validation...")

        trainer_config = TrainingConfig(
            max_iterations=config.max_iterations,
            improvement_threshold=config.improvement_threshold,
            random_state=args.seed,
            verbose=False
        )

        # Combine data for CV
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])

        trainer = QNNTrainer(model, trainer_config)
        cv_results = trainer.cross_validate(X_all, y_all, n_folds=args.cv)

        print(f"\nCross-Validation Results:")
        print(f"  Mean Accuracy: {cv_results['mean_val_accuracy']:.4f} "
              f"(+/- {cv_results['std_val_accuracy']*2:.4f})")

    # Comparison with baselines
    if args.compare:
        print("\n" + "="*60)
        print("Comparison with Baselines")
        print("="*60)

        from src.training.trainer import ComparisonTrainer

        comparison = ComparisonTrainer(
            n_qubits=config.n_qubits,
            n_classes=len(np.unique(y_train)),
            random_state=args.seed
        )

        comparison_results = comparison.run_comparison(
            X_train, y_train, X_test, y_test,
            methods=['adaptive', 'standard_cobyla']
        )

        print("\n" + comparison.generate_report())

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        from src.utils import save_model
        model_path = output_dir / f"{args.dataset}_model.pkl"
        save_model(model, str(model_path))
        print(f"\nModel saved to: {model_path}")

    if args.save_plots:
        import matplotlib.pyplot as plt

        # Training history
        history = model.get_training_history()
        fig = plot_training_history(history)
        plot_path = output_dir / f"{args.dataset}_training.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training plot saved to: {plot_path}")

        # Circuit diagram
        try:
            circuit_path = output_dir / f"{args.dataset}_circuit.png"
            circuit_fig = model.circuit.draw(output='mpl')
            circuit_fig.savefig(circuit_path, dpi=150, bbox_inches='tight')
            plt.close(circuit_fig)
            print(f"Circuit diagram saved to: {circuit_path}")
        except Exception as e:
            print(f"Could not save circuit diagram: {e}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60 + "\n")

    return model, test_metrics


if __name__ == '__main__':
    main()
