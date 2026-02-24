#!/usr/bin/env python3
"""
generate_results.py
===================
Generates comprehensive comparison graphs for Chapter 6: Results and Analysis.

Reads all comparison_results.json files from results/ and produces:
  1. Accuracy comparison bar chart (all datasets)
  2. Training cost convergence curves (per dataset)
  3. Circuit complexity comparison (gates, depth, params)
  4. Training time comparison
  5. Confusion matrices (per dataset)
  6. Combined performance summary table image
  7. F1 / Balanced Accuracy / AUC-ROC grouped bar chart
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'analysis_graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nice short names for display
DATASET_LABELS = {
    'mnist_0_1_7x7':     'MNIST 0v1\n(7×7)',
    'mnist_3_5_7x7':     'MNIST 3v5\n(7×7)',
    'mnist_pca_10d':     'MNIST PCA\n(10D)',
    'two-curves':        'Two\nCurves',
    'linearly-separable':'Linearly\nSeparable',
    'bars-and-stripes':  'Bars &\nStripes',
}

METHOD_COLORS = {
    'ARC_QNN':           '#2ecc71',   # green
    'Fixed_QNN_COBYLA':  '#e74c3c',   # red
}

METHOD_LABELS = {
    'ARC_QNN':           'ARC QNN',
    'Fixed_QNN_COBYLA':  'Fixed Ansatz (COBYLA)',
}

# Preferred dataset display order
DATASET_ORDER = [
    'two-curves', 'linearly-separable', 'bars-and-stripes',
    'mnist_0_1_7x7', 'mnist_3_5_7x7', 'mnist_pca_10d',
]

# ──────────────────────────────────────────────────────────────
# Load all results
# ──────────────────────────────────────────────────────────────
def load_all_results():
    """Load comparison_results.json from each dataset subfolder."""
    all_results = {}
    for ds in DATASET_ORDER:
        path = os.path.join(RESULTS_DIR, ds, 'comparison_results.json')
        if os.path.exists(path):
            with open(path) as f:
                all_results[ds] = json.load(f)
            print(f"  Loaded: {ds}")
        else:
            print(f"  MISSING: {ds}")
    return all_results


# ──────────────────────────────────────────────────────────────
# 1. Accuracy Comparison Bar Chart (all datasets)
# ──────────────────────────────────────────────────────────────
def plot_accuracy_comparison(all_results):
    """Bar chart: test accuracy for ARC vs Fixed across all datasets."""
    datasets = [d for d in DATASET_ORDER if d in all_results]
    labels   = [DATASET_LABELS.get(d, d) for d in datasets]

    arc_acc   = []
    fixed_acc = []
    for ds in datasets:
        methods = all_results[ds]['methods']
        arc_acc.append(methods['ARC_QNN']['test_metrics']['accuracy'] * 100)
        fixed_acc.append(methods['Fixed_QNN_COBYLA']['test_metrics']['accuracy'] * 100)

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, arc_acc,   width, label='ARC QNN',
                   color='#2ecc71', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, fixed_acc, width, label='Fixed Ansatz (COBYLA)',
                   color='#e74c3c', edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy: ARC QNN vs Fixed Ansatz (COBYLA)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4, label='Random baseline')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_1_accuracy_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 2. Training Cost Convergence Curves
# ──────────────────────────────────────────────────────────────
def plot_cost_curves(all_results):
    """Training cost convergence for each dataset (subplots).

    Uses dual Y-axes because ARC cost (log-loss 0.1–0.7) and Fixed cost
    (log-loss 1–6) live on very different scales.  ARC also has far fewer
    points (1–15 gate iterations) vs Fixed (100 COBYLA iterations).
    """
    datasets = [d for d in DATASET_ORDER if d in all_results]
    n = len(datasets)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    axes_flat = [ax for row in (axes if rows > 1 else [axes])
                 for ax in (row if hasattr(row, '__iter__') else [row])]

    for i, ds in enumerate(datasets):
        ax_left = axes_flat[i]
        methods = all_results[ds]['methods']

        arc_hist   = methods.get('ARC_QNN', {}).get('cost_history', [])
        fixed_hist = methods.get('Fixed_QNN_COBYLA', {}).get('cost_history', [])

        # ---- ARC on left y-axis (green) --------------------------------
        if arc_hist:
            x_arc = range(1, len(arc_hist) + 1)
            ln1 = ax_left.plot(x_arc, arc_hist,
                               color='#2ecc71', linewidth=2.5, marker='o',
                               markersize=7, label='ARC QNN', zorder=5)
            # Annotate start / end values
            ax_left.annotate(f'{arc_hist[0]:.3f}',
                             xy=(1, arc_hist[0]), fontsize=7,
                             textcoords='offset points', xytext=(8, 6),
                             color='#2ecc71', fontweight='bold')
            ax_left.annotate(f'{arc_hist[-1]:.3f}',
                             xy=(len(arc_hist), arc_hist[-1]), fontsize=7,
                             textcoords='offset points', xytext=(8, -10),
                             color='#2ecc71', fontweight='bold')

        ax_left.set_xlabel('Iteration / Gate Number', fontsize=9)
        ax_left.set_ylabel('ARC Cost (Log-Loss)', fontsize=10, color='#2ecc71')
        ax_left.tick_params(axis='y', labelcolor='#2ecc71')
        if arc_hist:
            y_min = min(arc_hist) * 0.9
            y_max = max(arc_hist) * 1.1
            ax_left.set_ylim(max(0, y_min), y_max)

        # ---- Fixed on right y-axis (red) --------------------------------
        ax_right = ax_left.twinx()
        if fixed_hist:
            x_fix = range(1, len(fixed_hist) + 1)
            ln2 = ax_right.plot(x_fix, fixed_hist,
                                color='#e74c3c', linewidth=1.5, alpha=0.7,
                                marker='.', markersize=2,
                                label='Fixed Ansatz (COBYLA)')
            ax_right.set_ylabel('Fixed Cost (Log-Loss)', fontsize=10, color='#e74c3c')
            ax_right.tick_params(axis='y', labelcolor='#e74c3c')

        # ---- Combined legend -------------------------------------------
        lines_labels = []
        for ax_tmp in [ax_left, ax_right]:
            h, l = ax_tmp.get_legend_handles_labels()
            lines_labels.extend(zip(h, l))
        if lines_labels:
            handles, labels = zip(*lines_labels)
            ax_left.legend(handles, labels, fontsize=8, loc='upper right')

        ax_left.set_title(DATASET_LABELS.get(ds, ds).replace('\n', ' '),
                          fontsize=11, fontweight='bold')
        ax_left.grid(alpha=0.3)

    # Hide unused axes
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Training Cost Convergence: ARC vs Fixed Ansatz\n'
                 '(Dual Y-Axes — different cost scales)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_2_cost_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")



# ──────────────────────────────────────────────────────────────
# 3. Circuit Complexity Comparison
# ──────────────────────────────────────────────────────────────
def plot_circuit_complexity(all_results):
    """Grouped bar chart: gates, depth, parameters for each dataset."""
    datasets = [d for d in DATASET_ORDER if d in all_results]
    labels   = [DATASET_LABELS.get(d, d) for d in datasets]

    metrics_to_plot = [
        ('n_gates',      'Number of Gates'),
        ('circuit_depth', 'Circuit Depth'),
        ('n_parameters',  'Number of Parameters'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (metric_key, metric_title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        arc_vals   = []
        fixed_vals = []
        for ds in datasets:
            methods = all_results[ds]['methods']
            arc_vals.append(methods['ARC_QNN'].get(metric_key, 0))
            fixed_vals.append(methods['Fixed_QNN_COBYLA'].get(metric_key, 0))

        x = np.arange(len(datasets))
        width = 0.35
        ax.bar(x - width/2, arc_vals,   width, label='ARC QNN',
               color='#2ecc71', edgecolor='white')
        ax.bar(x + width/2, fixed_vals, width, label='Fixed Ansatz',
               color='#e74c3c', edgecolor='white')

        # Value labels
        for j, (a, f_) in enumerate(zip(arc_vals, fixed_vals)):
            ax.text(j - width/2, a + 0.5, str(a), ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(j + width/2, f_ + 0.5, str(f_), ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(metric_title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Circuit Complexity: ARC QNN vs Fixed Ansatz', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_3_circuit_complexity.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 4. Training Time Comparison
# ──────────────────────────────────────────────────────────────
def plot_training_time(all_results):
    """Horizontal bar chart: training time for each method/dataset."""
    datasets = [d for d in DATASET_ORDER if d in all_results]
    labels   = [DATASET_LABELS.get(d, d).replace('\n', ' ') for d in datasets]

    arc_times   = []
    fixed_times = []
    for ds in datasets:
        methods = all_results[ds]['methods']
        arc_times.append(methods['ARC_QNN']['training_time'])
        fixed_times.append(methods['Fixed_QNN_COBYLA']['training_time'])

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.barh(x - width/2, arc_times,   width, label='ARC QNN',
                    color='#2ecc71', edgecolor='white')
    bars2 = ax.barh(x + width/2, fixed_times, width, label='Fixed Ansatz (COBYLA)',
                    color='#e74c3c', edgecolor='white')

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}s', ha='left', va='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}s', ha='left', va='center', fontsize=9)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time: ARC QNN vs Fixed Ansatz', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_4_training_time.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 5. Confusion Matrices
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrices(all_results):
    """Confusion matrices for ARC and Fixed on each dataset."""
    datasets = [d for d in DATASET_ORDER if d in all_results]
    n = len(datasets)

    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5*n))
    if n == 1:
        axes = [axes]

    for i, ds in enumerate(datasets):
        methods = all_results[ds]['methods']
        ds_label = DATASET_LABELS.get(ds, ds).replace('\n', ' ')

        for j, (method_name, method_label) in enumerate([
            ('ARC_QNN', 'ARC QNN'),
            ('Fixed_QNN_COBYLA', 'Fixed Ansatz')
        ]):
            ax = axes[i][j]
            cm = np.array(methods[method_name]['test_metrics']['confusion_matrix'])
            acc = methods[method_name]['test_metrics']['accuracy'] * 100

            cmap = 'Greens' if j == 0 else 'Reds'
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                        xticklabels=['Pred 0', 'Pred 1'],
                        yticklabels=['True 0', 'True 1'],
                        cbar=False, annot_kws={'size': 12})
            ax.set_title(f'{ds_label} — {method_label}\nAcc: {acc:.1f}%',
                         fontsize=10, fontweight='bold')

    fig.suptitle('Confusion Matrices: Test Set Predictions', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_5_confusion_matrices.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 6. Performance Summary Table (as image)
# ──────────────────────────────────────────────────────────────
def plot_summary_table(all_results):
    """Render the performance comparison as a styled table image."""
    datasets = [d for d in DATASET_ORDER if d in all_results]

    # Build table data
    columns = ['Dataset', 'Qubits',
               'ARC\nAccuracy', 'Fixed\nAccuracy',
               'ARC\nGates', 'Fixed\nParams',
               'ARC\nDepth', 'Fixed\nDepth',
               'ARC\nTime(s)', 'Fixed\nTime(s)', 'Improvement']
    rows = []
    for ds in datasets:
        info = all_results[ds]['dataset_info']
        arc  = all_results[ds]['methods']['ARC_QNN']
        fix  = all_results[ds]['methods']['Fixed_QNN_COBYLA']

        arc_acc = arc['test_metrics']['accuracy'] * 100
        fix_acc = fix['test_metrics']['accuracy'] * 100
        improvement = arc_acc - fix_acc

        rows.append([
            DATASET_LABELS.get(ds, ds).replace('\n', ' '),
            str(info['n_qubits']),
            f'{arc_acc:.1f}%',
            f'{fix_acc:.1f}%',
            str(arc.get('n_gates', '-')),
            str(fix.get('n_parameters', '-')),
            str(arc.get('circuit_depth', '-')),
            str(fix.get('circuit_depth', '-')),
            f'{arc["training_time"]:.1f}',
            f'{fix["training_time"]:.1f}',
            f'+{improvement:.1f}%',
        ])

    fig, ax = plt.subplots(figsize=(16, 1.2 + 0.6*len(rows)))
    ax.axis('off')

    table = ax.table(
        cellText=rows, colLabels=columns,
        cellLoc='center', loc='center',
        colColours=['#3498db']*len(columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#2c3e50')

    # Color improvement column and accuracy columns
    for i in range(len(rows)):
        # ARC accuracy in green shading
        arc_cell = table[i+1, 2]
        fix_cell = table[i+1, 3]
        imp_cell = table[i+1, 10]

        arc_val = float(rows[i][2].replace('%', ''))
        fix_val = float(rows[i][3].replace('%', ''))

        if arc_val > fix_val:
            arc_cell.set_facecolor('#d4efdf')
            fix_cell.set_facecolor('#fadbd8')
            imp_cell.set_facecolor('#d4efdf')
            imp_cell.set_text_props(fontweight='bold', color='#27ae60')
        else:
            arc_cell.set_facecolor('#fadbd8')
            fix_cell.set_facecolor('#d4efdf')

        # Alternate row colors
        base_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in [0, 1, 4, 5, 6, 7, 8, 9]:
            table[i+1, j].set_facecolor(base_color)

    ax.set_title('Performance Comparison: ARC QNN vs Fixed Ansatz (COBYLA)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_6_performance_table.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 7. Multi-metric Comparison (F1, Balanced Acc, AUC-ROC)
# ──────────────────────────────────────────────────────────────
def plot_multi_metric(all_results):
    """Grouped bar chart for F1, Balanced Accuracy, AUC-ROC across datasets."""
    datasets = [d for d in DATASET_ORDER if d in all_results]
    labels   = [DATASET_LABELS.get(d, d) for d in datasets]

    metric_keys = [
        ('f1_score',          'F1 Score'),
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('auc_roc',           'AUC-ROC'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (metric_key, metric_title) in enumerate(metric_keys):
        ax = axes[idx]
        arc_vals   = []
        fixed_vals = []
        for ds in datasets:
            methods = all_results[ds]['methods']
            arc_vals.append(methods['ARC_QNN']['test_metrics'].get(metric_key, 0) * 100)
            fixed_vals.append(methods['Fixed_QNN_COBYLA']['test_metrics'].get(metric_key, 0) * 100)

        x = np.arange(len(datasets))
        width = 0.35
        ax.bar(x - width/2, arc_vals,   width, label='ARC QNN',
               color='#2ecc71', edgecolor='white')
        ax.bar(x + width/2, fixed_vals, width, label='Fixed Ansatz',
               color='#e74c3c', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f'{metric_title} (%)', fontsize=10)
        ax.set_title(metric_title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4)

    fig.suptitle('Classification Metrics: ARC QNN vs Fixed Ansatz', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_7_multi_metrics.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 8. Train vs Test Accuracy (overfitting check)
# ──────────────────────────────────────────────────────────────
def plot_train_test_comparison(all_results):
    """Show train vs test accuracy for ARC to check overfitting."""
    datasets = [d for d in DATASET_ORDER if d in all_results]
    labels   = [DATASET_LABELS.get(d, d).replace('\n', ' ') for d in datasets]

    arc_train = []
    arc_test  = []
    fix_train = []
    fix_test  = []
    for ds in datasets:
        methods = all_results[ds]['methods']
        arc_train.append(methods['ARC_QNN']['train_metrics']['accuracy'] * 100)
        arc_test.append(methods['ARC_QNN']['test_metrics']['accuracy'] * 100)
        fix_train.append(methods['Fixed_QNN_COBYLA']['train_metrics']['accuracy'] * 100)
        fix_test.append(methods['Fixed_QNN_COBYLA']['test_metrics']['accuracy'] * 100)

    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, arc_train, width, label='ARC Train', color='#27ae60', alpha=0.7)
    ax.bar(x - 0.5*width, arc_test,  width, label='ARC Test',  color='#2ecc71')
    ax.bar(x + 0.5*width, fix_train, width, label='Fixed Train', color='#c0392b', alpha=0.7)
    ax.bar(x + 1.5*width, fix_test,  width, label='Fixed Test',  color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Train vs Test Accuracy (Overfitting Analysis)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_8_train_test_accuracy.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Generating Results & Analysis Graphs")
    print("=" * 60)

    print("\nLoading results...")
    all_results = load_all_results()

    if not all_results:
        print("ERROR: No results found in results/ directory!")
        sys.exit(1)

    print(f"\nFound {len(all_results)} datasets with results.")

    print("\nGenerating graphs...")
    plot_accuracy_comparison(all_results)
    plot_cost_curves(all_results)
    plot_circuit_complexity(all_results)
    plot_training_time(all_results)
    plot_confusion_matrices(all_results)
    plot_summary_table(all_results)
    plot_multi_metric(all_results)
    plot_train_test_comparison(all_results)

    print(f"\n{'=' * 60}")
    print(f"All graphs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # Print text summary
    print("\n\nPERFORMANCE SUMMARY TABLE")
    print("-" * 90)
    print(f"{'Dataset':<22} {'Qubits':>6} {'ARC Acc':>8} {'Fix Acc':>8} "
          f"{'ARC Gates':>9} {'Fix Params':>10} {'ARC Time':>9} {'Fix Time':>9}")
    print("-" * 90)
    for ds in DATASET_ORDER:
        if ds not in all_results:
            continue
        info = all_results[ds]['dataset_info']
        arc  = all_results[ds]['methods']['ARC_QNN']
        fix  = all_results[ds]['methods']['Fixed_QNN_COBYLA']
        print(f"{ds:<22} {info['n_qubits']:>6} "
              f"{arc['test_metrics']['accuracy']*100:>7.1f}% "
              f"{fix['test_metrics']['accuracy']*100:>7.1f}% "
              f"{arc.get('n_gates', '-'):>9} "
              f"{fix.get('n_parameters', '-'):>10} "
              f"{arc['training_time']:>8.1f}s "
              f"{fix['training_time']:>8.1f}s")
    print("-" * 90)


if __name__ == '__main__':
    main()
