#!/usr/bin/env python3
"""
run_all_datasets.py
===================
Run the ARC-vs-Fixed QNN comparison on every CSV dataset in the
Datasets/ folder.  Results are saved per-dataset under results/<dataset_name>/.

Usage
-----
Quick run (small settings for testing):
    python run_all_datasets.py --quick

Full run (default):
    python run_all_datasets.py

Custom qubits or gates:
    python run_all_datasets.py --n_qubits 6 --arc_max_gates 40

With IBM hardware:
    python run_all_datasets.py --ibm
"""

import argparse
import os
import sys
import subprocess
import time
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset name -> (recommended qubits, description)
DATASETS = {
    'two-curves':          (4,  '2-class synthetic curves, 11 features'),
    'linearly-separable':  (4,  '2-class linear, 11 features'),
    'bars-and-stripes':    (4,  'Bars & stripes 4-dim, 17 features'),
    'mnist_0_1_7x7':       (4,  'MNIST 0-vs-1 (7x7), 50 features'),
    'mnist_0_1_14x14':     (4,  'MNIST 0-vs-1 (14x14), 197 features'),
    'mnist_0_1_28x28':     (4,  'MNIST 0-vs-1 (28x28), 785 features'),
    'mnist_3_5_7x7':       (4,  'MNIST 3-vs-5 (7x7), 50 features'),
    'mnist_3_5_14x14':     (4,  'MNIST 3-vs-5 (14x14), 197 features'),
    'mnist_3_5_28x28':     (4,  'MNIST 3-vs-5 (28x28), 785 features'),
    'mnist_pca_10d':       (4,  'MNIST PCA 10-dim, 11 features'),
}

def main():
    parser = argparse.ArgumentParser(
        description='Run ARC vs Fixed QNN comparison on all CSV datasets'
    )
    parser.add_argument('--n_qubits', type=int, default=None,
                        help='Override qubit count for ALL datasets '
                             '(default: use per-dataset recommendation)')
    parser.add_argument('--arc_max_gates', type=int, default=30,
                        help='Max gates for ARC (default: 30)')
    parser.add_argument('--fixed_layers', type=int, default=3,
                        help='Layers for fixed ansatz (default: 3)')
    parser.add_argument('--fixed_max_iter', type=int, default=100,
                        help='Max optimizer iterations (default: 100)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['arc', 'cobyla'],
                        help='Methods to compare (default: arc cobyla)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--ibm', action='store_true',
                        help='Also run on IBM hardware (uses IBM_TOKEN from .env)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer gates/iterations for testing')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific datasets to run (default: all). '
                             f'Available: {list(DATASETS.keys())}')
    args = parser.parse_args()

    if args.quick:
        args.arc_max_gates = 5
        args.fixed_layers = 1
        args.fixed_max_iter = 10

    datasets_to_run = args.datasets if args.datasets else list(DATASETS.keys())

    # Validate dataset names
    for ds in datasets_to_run:
        if ds not in DATASETS:
            print(f"ERROR: Unknown dataset '{ds}'")
            print(f"Available: {list(DATASETS.keys())}")
            sys.exit(1)

    print(f"{'='*70}")
    print(f"  ARC vs Fixed QNN - Batch Run on {len(datasets_to_run)} datasets")
    print(f"  ARC max gates: {args.arc_max_gates}  |  "
          f"Fixed layers: {args.fixed_layers}  |  "
          f"Fixed max iter: {args.fixed_max_iter}")
    print(f"  Methods: {args.methods}")
    print(f"  IBM hardware: {'YES' if args.ibm else 'no'}")
    print(f"{'='*70}")

    results_summary = {}
    total_time_start = time.time()

    for i, ds_name in enumerate(datasets_to_run, 1):
        rec_qubits, desc = DATASETS[ds_name]
        nq = args.n_qubits if args.n_qubits else rec_qubits
        out_dir = os.path.join(args.output_dir, ds_name)

        print(f"\n[{i}/{len(datasets_to_run)}] {ds_name} "
              f"({desc}, n_qubits={nq})")
        print('-' * 60)

        cmd = [
            sys.executable, os.path.join(ROOT, 'compare_qnn.py'),
            '--dataset', ds_name,
            '--n_qubits', str(nq),
            '--arc_max_gates', str(args.arc_max_gates),
            '--fixed_layers', str(args.fixed_layers),
            '--fixed_max_iter', str(args.fixed_max_iter),
            '--methods', *args.methods,
            '--seed', str(args.seed),
            '--output_dir', out_dir,
        ]
        if args.ibm:
            # Token will be picked up from .env automatically
            ibm_token = os.environ.get('IBM_TOKEN')
            if ibm_token:
                cmd += ['--ibm_token', ibm_token]
            else:
                print("  WARNING: --ibm flag set but IBM_TOKEN not found in .env")

        t0 = time.time()
        ret = subprocess.run(cmd)
        elapsed = time.time() - t0

        # Read results JSON if available
        json_path = os.path.join(out_dir, 'comparison_results.json')
        accuracy_info = ''
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    res = json.load(f)
                accs = []
                for m, r in res.get('methods', {}).items():
                    acc = r.get('test_metrics', {}).get('accuracy', '?')
                    accs.append(f"{m}={acc}")
                accuracy_info = ', '.join(accs)
            except Exception:
                pass

        status = 'OK' if ret.returncode == 0 else f'FAIL (code {ret.returncode})'
        results_summary[ds_name] = {
            'status': status,
            'time': elapsed,
            'accuracy': accuracy_info,
        }
        print(f"  -> {status} ({elapsed:.1f}s) {accuracy_info}")

    total_time = time.time() - total_time_start

    # ---- Summary table -------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  BATCH SUMMARY  ({total_time:.1f}s total)")
    print(f"{'='*70}")
    print(f"  {'Dataset':<30s} {'Status':<12s} {'Time':>8s}  Accuracy")
    print(f"  {'-'*30} {'-'*12} {'-'*8}  {'-'*30}")
    for ds_name, info in results_summary.items():
        print(f"  {ds_name:<30s} {info['status']:<12s} "
              f"{info['time']:>7.1f}s  {info['accuracy']}")
    print(f"  {'-'*30} {'-'*12} {'-'*8}")
    print(f"  Total: {total_time:.1f}s")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'batch_summary.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
