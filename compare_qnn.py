#!/usr/bin/env python3
"""
compare_qnn.py
==============

Main comparison script:
  - ARC QNN  (Analytic Iterative Circuit Reconstruction – dynamic ansatz)
  - Fixed Ansatz QNN  (static VQC with COBYLA / SPSA)

Supports multiple datasets, barren-plateau analysis, and optional
execution on real IBM Quantum hardware.

Usage examples
--------------
Quick test on moons:
    python compare_qnn.py --dataset moons --n_qubits 2

Reference two-curves:
    python compare_qnn.py --dataset two-curves --n_qubits 4

MNIST 0-vs-1 (7×7 PCA to n_qubits features):
    python compare_qnn.py --dataset mnist_0_1_7x7 --n_qubits 4

Full comparison with IBM hardware:
    python compare_qnn.py --dataset moons --n_qubits 4 --ibm_token YOUR_TOKEN

Barren-plateau analysis (variance of cost across random inits):
    python compare_qnn.py --dataset moons --n_qubits 4 --barren_plateau --bp_trials 20
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import warnings
# Suppress noisy library warnings but keep our own print-based messages
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---- Load .env file ---------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; user can still set env vars manually

# ---- project root on sys.path -----------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =============================================================================
# Dataset helpers
# =============================================================================

# All reference-style CSV datasets
REFERENCE_DATASETS = {
    'two-curves':        ('two-curves_train_1000',       'two-curves_test_1000'),
    'linearly-separable':('linearly-separable_train_1000','linearly-separable_test_1000'),
    'bars-and-stripes':  ('bars-and-stripes_4d_1000',    'bars-and-stripes_4d_200'),
    'mnist_0_1_7x7':     ('mnist_train_0_1_7x7_N_1000',  'mnist_test_0_1_7x7_N_1000'),
    'mnist_0_1_14x14':   ('mnist_train_0_1_14x14_N_1000', 'mnist_test_0_1_14x14_N_1000'),
    'mnist_0_1_28x28':   ('mnist_train_0_1_28x28_N_1000', 'mnist_test_0_1_28x28_N_1000'),
    'mnist_3_5_7x7':     ('mnist_train_3_5_7x7_N_1000',  'mnist_test_3_5_7x7_N_1000'),
    'mnist_3_5_14x14':   ('mnist_train_3_5_14x14_N_1000', 'mnist_test_3_5_14x14_N_1000'),
    'mnist_3_5_28x28':   ('mnist_train_3_5_28x28_N_1000', 'mnist_test_3_5_28x28_N_1000'),
    'mnist_pca_10d':     ('mnist-pca_train_10d_11552',   'mnist-pca_test_10d_1902'),
}

def _find_data_dir():
    """Return the first existing Datasets directory."""
    candidates = [
        os.path.join(ROOT, 'Datasets'),
        os.path.join(ROOT, 'Analytic-QNN-Reconstruction', 'Datasets'),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"Cannot find Datasets folder. Searched:\n  "
        + "\n  ".join(candidates)
    )


def load_reference_dataset(name: str, n_qubits: int):
    """Load a dataset from the CSV files in the Datasets folder.

    If n_qubits == 0 (auto mode), returns all features (PCA capping handled by caller).
    Otherwise PCA is applied when features > n_qubits.
    """
    data_dir = _find_data_dir()
    dataset_map = REFERENCE_DATASETS

    if name not in dataset_map:
        raise ValueError(f"Unknown reference dataset: {name}. "
                         f"Available: {list(dataset_map.keys())}")

    train_tag, test_tag = dataset_map[name]
    X_train = np.loadtxt(os.path.join(data_dir, f'X{train_tag}.csv'), delimiter=',')
    y_train = np.loadtxt(os.path.join(data_dir, f'Y{train_tag}.csv'), delimiter=',')
    X_test  = np.loadtxt(os.path.join(data_dir, f'X{test_tag}.csv'), delimiter=',')
    y_test  = np.loadtxt(os.path.join(data_dir, f'Y{test_tag}.csv'), delimiter=',')

    # n_qubits == 0 means auto: n_qubit = n_feature (reference behaviour)
    if n_qubits > 0 and X_train.shape[1] > n_qubits:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_qubits)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        print(f"  PCA: {X_train.shape[1]+n_qubits} -> {n_qubits} features "
              f"(explained variance {pca.explained_variance_ratio_.sum():.2%})")
    else:
        print(f"  Using all {X_train.shape[1]} features (n_qubit = n_feature)")

    # Ensure binary labels 0/1
    unique_labels = np.unique(y_train)
    if set(unique_labels) == {-1, 1}:
        y_train = np.where(y_train == -1, 0, 1).astype(int)
        y_test  = np.where(y_test  == -1, 0, 1).astype(int)
    elif set(unique_labels) != {0, 1}:
        # Map to 0/1
        y_train = (y_train == unique_labels[1]).astype(int)
        y_test  = (y_test  == unique_labels[1]).astype(int)

    return X_train, X_test, y_train.astype(int), y_test.astype(int)


def load_sklearn_dataset(name: str, n_qubits: int, n_samples: int = 300):
    """Load a scikit-learn dataset."""
    from src.data.datasets import (
        load_moons_quantum, load_iris_quantum,
        load_circles_quantum, load_breast_cancer_quantum
    )

    if name == 'moons':
        return load_moons_quantum(n_samples=n_samples, n_qubits=n_qubits)
    elif name == 'iris':
        return load_iris_quantum(n_qubits=n_qubits)
    elif name == 'circles':
        return load_circles_quantum(n_samples=n_samples, n_qubits=n_qubits)
    elif name == 'breast_cancer':
        return load_breast_cancer_quantum(n_qubits=n_qubits)
    else:
        raise ValueError(f"Unknown sklearn dataset: {name}")


# =============================================================================
# Barren-plateau analysis
# =============================================================================

def barren_plateau_analysis(
    X_train, y_train, n_qubits: int, n_layers_list=None,
    n_trials: int = 10, verbose: bool = True
):
    """
    Analyse barren plateaus by measuring the *variance* of the initial
    cost function across many random parameter initialisations for the
    fixed-ansatz QNN at different circuit depths.

    For the ARC method there is no random initialisation (each gate angle
    is determined analytically), so we measure the variance of the final
    cost across different random seeds / data shuffles instead.

    Returns
    -------
    dict with keys 'fixed_ansatz' and 'arc'.
    """
    from src.models.fixed_ansatz_qnn import FixedAnsatzQNN
    from sklearn.metrics import log_loss

    if n_layers_list is None:
        n_layers_list = [1, 2, 3, 5, 8]

    results = {'fixed_ansatz': {}, 'arc': {}}

    # --- Fixed ansatz: variance of initial cost across random inits ----------
    for n_layers in n_layers_list:
        costs = []
        for trial in range(n_trials):
            model = FixedAnsatzQNN(
                n_qubits=n_qubits, n_layers=n_layers,
                optimizer='cobyla', max_iter=0, verbose=False
            )
            # Initialise randomly and evaluate cost once
            model._build_circuit(X_train.shape[1])
            rng = np.random.RandomState(trial)
            random_params = rng.uniform(0, 2*np.pi, model.n_variational_params)

            from qiskit.quantum_info import Statevector
            total_cost = 0.0
            for xi, yi in zip(X_train, y_train):
                param_dict = {}
                for p, v in zip(model.data_params, xi[:n_qubits]):
                    param_dict[p] = float(v)
                for p, v in zip(model.var_params, random_params):
                    param_dict[p] = float(v)
                sv = Statevector.from_instruction(model.circuit.assign_parameters(param_dict))
                prob_0 = abs(sv[0]) ** 2
                prob_0 = np.clip(prob_0, 1e-15, 1 - 1e-15)
                total_cost += -(yi * np.log(prob_0) + (1 - yi) * np.log(1 - prob_0))
            costs.append(total_cost / len(X_train))

        results['fixed_ansatz'][n_layers] = {
            'mean': float(np.mean(costs)),
            'variance': float(np.var(costs)),
            'std': float(np.std(costs)),
            'costs': [float(c) for c in costs],
        }
        if verbose:
            print(f"  Fixed ansatz (L={n_layers}): mean={np.mean(costs):.4f}, "
                  f"var={np.var(costs):.6f}")

    # --- ARC: variance of final cost across seeds ---------------------------
    from src.estimators.arc_estimator import ARCEstimator
    from qiskit.circuit import ParameterVector

    arc_costs = []
    for trial in range(min(n_trials, 5)):  # ARC is slower, limit trials
        params = ParameterVector('x', X_train.shape[1])
        estimator = ARCEstimator(
            n_qubits=n_qubits,
            gate_list=['U1', 'U2', 'U3', 'H', 'X', 'Z'],
            max_gates=20, verbose=False
        )
        np.random.seed(trial)
        idx = np.random.permutation(len(X_train))
        _, _, cost_history = estimator.reconstruct_circuit(
            X_train[idx], y_train[idx], params
        )
        arc_costs.append(cost_history[-1] if cost_history else float('inf'))

    results['arc'] = {
        'mean': float(np.mean(arc_costs)),
        'variance': float(np.var(arc_costs)),
        'std': float(np.std(arc_costs)),
        'costs': [float(c) for c in arc_costs],
    }
    if verbose:
        print(f"  ARC QNN: mean={np.mean(arc_costs):.4f}, "
              f"var={np.var(arc_costs):.6f}")

    return results


def plot_barren_plateau(bp_results, save_path=None):
    """Visualise the barren-plateau analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping barren-plateau plot")
        return

    fixed = bp_results['fixed_ansatz']
    layers = sorted(fixed.keys())
    variances = [fixed[l]['variance'] for l in layers]
    means = [fixed[l]['mean'] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Variance vs depth
    axes[0].semilogy(layers, variances, 'o-', label='Fixed Ansatz')
    arc_var = bp_results['arc']['variance']
    axes[0].axhline(y=arc_var, color='r', linestyle='--', label=f'ARC (var={arc_var:.6f})')
    axes[0].set_xlabel('Number of Layers')
    axes[0].set_ylabel('Variance of Initial Cost')
    axes[0].set_title('Barren Plateau Analysis')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mean cost vs depth
    axes[1].plot(layers, means, 'o-', label='Fixed Ansatz (random init)')
    axes[1].axhline(y=bp_results['arc']['mean'], color='r', linestyle='--',
                    label=f'ARC (final cost)')
    axes[1].set_xlabel('Number of Layers')
    axes[1].set_ylabel('Mean Cost (Log-Loss)')
    axes[1].set_title('Cost vs Circuit Depth')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_path or 'results/barren_plateau_analysis.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Barren-plateau plot saved: {path}")


def plot_comparison(results, save_path=None):
    """Plot training cost curves for all methods side-by-side."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping comparison plot")
        return

    methods = results['methods']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cost curves
    for name, r in methods.items():
        history = r.get('cost_history', [])
        if history:
            axes[0].plot(history, label=name)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Cost (Log-Loss)')
    axes[0].set_title('Training Cost Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Summary bar chart
    metric_names = ['accuracy', 'balanced_accuracy', 'f1_score']
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(methods), 1)
    for i, (name, r) in enumerate(methods.items()):
        vals = [r['test_metrics'].get(m, 0) for m in metric_names]
        axes[1].bar(x + i * width, vals, width, label=name)
    axes[1].set_xticks(x + width * (len(methods) - 1) / 2)
    axes[1].set_xticklabels(['Accuracy', 'Balanced\nAccuracy', 'F1'])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Test Set Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = save_path or 'results/comparison.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Comparison plot saved: {path}")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare ARC QNN (dynamic) vs Fixed Ansatz QNN (static)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Data
    parser.add_argument('--dataset', type=str, default='moons',
                        help='Dataset name. sklearn: moons, iris, circles, '
                             'breast_cancer. CSV: two-curves, linearly-separable, '
                             'bars-and-stripes, mnist_0_1_7x7, mnist_0_1_14x14, '
                             'mnist_0_1_28x28, mnist_3_5_7x7, mnist_3_5_14x14, '
                             'mnist_3_5_28x28, mnist_pca_10d. '
                             'Use "all" to run every CSV dataset.')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of samples for synthetic datasets')
    parser.add_argument('--n_qubits', type=int, default=0,
                        help='Number of qubits (0 = auto: uses n_feature, capped at --max_qubits)')
    parser.add_argument('--max_qubits', type=int, default=20,
                        help='Maximum qubits in auto mode (PCA applied if features exceed this)')

    # ARC settings
    parser.add_argument('--arc_max_gates', type=int, default=15,
                        help='Maximum gates for ARC construction')
    parser.add_argument('--arc_gate_list', type=str, nargs='+',
                        default=['U1', 'U2', 'U3', 'H', 'X', 'Z'],
                        help='Gate types for ARC')
    parser.add_argument('--arc_subsample', type=int, default=100,
                        help='Max training samples per ARC gate iteration (0=all, default 100)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Parallel jobs for joblib (-1 = all cores, 1 = sequential)')
    parser.add_argument('--patience', type=int, default=1,
                        help='Number of consecutive non-improving gates before stopping ARC '
                             '(1=original, 2=allow 1 extra chance, etc.)')

    # Fixed ansatz settings
    parser.add_argument('--fixed_layers', type=int, default=3,
                        help='Number of layers for fixed ansatz')
    parser.add_argument('--fixed_max_iter', type=int, default=100,
                        help='Max optimizer iterations for fixed ansatz')

    # Comparison
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['arc', 'cobyla'],
                        help='Methods to compare: arc, cobyla, spsa, nelder-mead, powell')

    # IBM hardware
    parser.add_argument('--ibm_token', type=str, default=None,
                        help='IBM Quantum API token. When provided the trained '
                             'circuits are also evaluated on real IBM hardware.')
    parser.add_argument('--ibm_backend', type=str, default='auto',
                        help='IBM backend name, or "auto" to pick the least-busy '
                             'backend automatically (default: auto)')
    parser.add_argument('--ibm_channel', type=str, default='ibm_cloud',
                        help='IBM channel: ibm_cloud or ibm_quantum_platform')
    parser.add_argument('--shots', type=int, default=4096,
                        help='Shots for hardware execution')

    # Barren-plateau analysis
    parser.add_argument('--barren_plateau', action='store_true',
                        help='Run barren-plateau analysis')
    parser.add_argument('--bp_trials', type=int, default=10,
                        help='Number of trials for barren-plateau analysis')
    parser.add_argument('--bp_layers', type=int, nargs='+', default=[1, 2, 3, 5, 8],
                        help='Layer counts for barren-plateau analysis')

    # General
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/<dataset>)')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Auto-set output_dir from dataset name if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join('results', args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    verbose = not args.quiet

    # ---- Handle "all" datasets -----------------------------------------------
    if args.dataset == 'all':
        print(f"\n{'='*60}")
        print("BATCH RUN: all CSV datasets")
        print(f"{'='*60}")
        # sensible qubit defaults per dataset
        qubit_map = {
            'two-curves': 4, 'linearly-separable': 4,
            'bars-and-stripes': 4,
            'mnist_0_1_7x7': 4, 'mnist_0_1_14x14': 4, 'mnist_0_1_28x28': 4,
            'mnist_3_5_7x7': 4, 'mnist_3_5_14x14': 4, 'mnist_3_5_28x28': 4,
            'mnist_pca_10d': 4,
        }
        summary = {}
        for ds_name in REFERENCE_DATASETS:
            nq = args.n_qubits if args.n_qubits != 4 else qubit_map.get(ds_name, 4)
            out_dir = os.path.join(args.output_dir, ds_name)
            cmd_args = [
                sys.executable, __file__,
                '--dataset', ds_name,
                '--n_qubits', str(nq),
                '--arc_max_gates', str(args.arc_max_gates),
                '--fixed_layers', str(args.fixed_layers),
                '--fixed_max_iter', str(args.fixed_max_iter),
                '--methods', *args.methods,
                '--seed', str(args.seed),
                '--output_dir', out_dir,
                '--n_samples', str(args.n_samples),
            ]
            if args.ibm_token:
                cmd_args += ['--ibm_token', args.ibm_token]
            if args.quiet:
                cmd_args.append('--quiet')
            print(f"\n>>> Running: {ds_name} (n_qubits={nq}) ...")
            import subprocess
            ret = subprocess.run(cmd_args)
            status = 'OK' if ret.returncode == 0 else f'FAIL (code {ret.returncode})'
            summary[ds_name] = status
            print(f"    {ds_name}: {status}")

        # print summary
        print(f"\n{'='*60}")
        print("BATCH SUMMARY")
        print(f"{'='*60}")
        for ds_name, status in summary.items():
            print(f"  {ds_name:30s} {status}")
        return

    # ---- Load data -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}  |  Qubits: {args.n_qubits if args.n_qubits > 0 else f'auto (max {args.max_qubits})'}")
    print(f"{'='*60}")

    ref_datasets = list(REFERENCE_DATASETS.keys())
    sklearn_datasets = ['moons', 'iris', 'circles', 'breast_cancer']

    if args.dataset in ref_datasets:
        X_train, X_test, y_train, y_test = load_reference_dataset(
            args.dataset, args.n_qubits
        )
    elif args.dataset in sklearn_datasets:
        X_train, X_test, y_train, y_test = load_sklearn_dataset(
            args.dataset, args.n_qubits if args.n_qubits > 0 else 4, args.n_samples
        )
    else:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {ref_datasets + sklearn_datasets}")
        sys.exit(1)

    # Auto-set n_qubits from actual feature count, capped at max_qubits
    if args.n_qubits == 0:
        n_feat = X_train.shape[1]
        if n_feat <= args.max_qubits:
            args.n_qubits = n_feat
            print(f"  Auto n_qubits = {args.n_qubits} (n_qubit = n_feature)")
        else:
            args.n_qubits = args.max_qubits
            print(f"  Auto n_qubits = {args.n_qubits} (capped from {n_feat} features, applying PCA)")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=args.n_qubits)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            print(f"  PCA: {n_feat} -> {args.n_qubits} features "
                  f"(explained variance {pca.explained_variance_ratio_.sum():.2%})")

    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ---- Run comparison ------------------------------------------------------
    from src.evaluation.comparison import QNNComparisonPipeline

    pipeline = QNNComparisonPipeline(
        n_qubits=args.n_qubits,
        arc_gate_list=args.arc_gate_list,
        arc_max_gates=args.arc_max_gates,
        fixed_layers=args.fixed_layers,
        fixed_max_iter=args.fixed_max_iter,
        seed=args.seed,
        verbose=verbose,
        arc_subsample_size=args.arc_subsample,
        n_jobs=args.n_jobs,
        patience=args.patience
    )

    results = pipeline.run_comparison(
        X_train, y_train, X_test, y_test,
        methods=args.methods,
        preprocess=True
    )

    # ---- IBM hardware (optional) ---------------------------------------------
    # Token: CLI arg > IBM_TOKEN env var
    # ibm_token = args.ibm_token or os.environ.get('IBM_TOKEN')
    ibm_token = args.ibm_token
    if ibm_token:
        print(f"\n{'='*60}")
        print(f"Running on IBM hardware (backend={args.ibm_backend})")
        print(f"{'='*60}")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, np.pi), clip=True)
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = pipeline.run_on_hardware(
            results, X_test_scaled, y_test,
            backend_name=args.ibm_backend,
            shots=args.shots,
            ibm_token=ibm_token,
            channel=args.ibm_channel
        )

    # ---- Barren-plateau analysis (optional) ----------------------------------
    if args.barren_plateau:
        print(f"\n{'='*60}")
        print(f"Barren Plateau Analysis ({args.bp_trials} trials)")
        print(f"{'='*60}")

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, np.pi), clip=True)
        X_train_scaled = scaler.fit_transform(X_train)

        bp_results = barren_plateau_analysis(
            X_train_scaled, y_train, args.n_qubits,
            n_layers_list=args.bp_layers,
            n_trials=args.bp_trials,
            verbose=verbose
        )
        results['barren_plateau'] = bp_results

        plot_barren_plateau(
            bp_results,
            save_path=os.path.join(args.output_dir, 'barren_plateau_analysis.png')
        )

    # ---- Report & save -------------------------------------------------------
    report = pipeline.print_report(results)
    plot_comparison(results, os.path.join(args.output_dir, 'comparison.png'))

    # Save text report
    report_path = os.path.join(args.output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Save JSON
    json_path = os.path.join(args.output_dir, 'comparison_results.json')
    pipeline.save_results(results, json_path)

    # ---- Print circuits ------------------------------------------------------
    import sys, io
    # Force UTF-8 stdout so Unicode box-drawing chars render on Windows
    utf8_out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                errors='replace', line_buffering=True)
    try:
        utf8_out.write(f"\n{'='*60}\n")
        utf8_out.write("QUANTUM CIRCUITS\n")
        utf8_out.write(f"{'='*60}\n")
        for method_name, method_result in results['methods'].items():
            circ = method_result.get('circuit')
            if circ is not None:
                utf8_out.write(f"\n--- {method_name} (depth={circ.depth()}, "
                               f"gates={circ.size()}) ---\n\n")
                utf8_out.write(str(circ.draw(output='text', fold=120)) + "\n")
                # Also save circuit diagram to file
                circ_path = os.path.join(args.output_dir,
                                         f'circuit_{method_name}.txt')
                with open(circ_path, 'w', encoding='utf-8') as cf:
                    cf.write(str(circ.draw(output='text', fold=120)))
                utf8_out.write(f"  (saved to {circ_path})\n")
        utf8_out.flush()
    finally:
        utf8_out.detach()  # prevent closing underlying stdout

    print(f"\nAll results saved to: {args.output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
