"""
Visualization Tools for Quantum Neural Networks
=================================================

Tools for visualizing QNN training, circuits, and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import warnings


def plot_training_history(
    history: List[Dict[str, Any]],
    metrics: List[str] = ['cost'],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history curves.

    Args:
        history: List of training history dictionaries
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [h.get(metric, h.get('cost')) for h in history]
        iterations = range(len(values))

        ax.plot(iterations, values, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Training {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_circuit(
    circuit,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    style: str = 'default'
) -> plt.Figure:
    """
    Plot quantum circuit diagram.

    Args:
        circuit: Qiskit QuantumCircuit
        figsize: Figure size
        save_path: Optional save path
        style: Circuit style

    Returns:
        Matplotlib figure
    """
    try:
        fig = circuit.draw(output='mpl', style=style, figsize=figsize)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
    except Exception as e:
        warnings.warn(f"Could not plot circuit: {e}")
        return None


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Tuple[int, int] = (0, 1),
    resolution: int = 50,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot decision boundary for 2D projection.

    Args:
        model: Trained QNN model
        X: Feature data
        y: Labels
        feature_indices: Which features to use for x and y axes
        resolution: Grid resolution
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get feature ranges
    i, j = feature_indices
    x_min, x_max = X[:, i].min() - 0.5, X[:, i].max() + 0.5
    y_min, y_max = X[:, j].min() - 0.5, X[:, j].max() + 0.5

    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Create input for prediction (fill other features with mean)
    n_features = X.shape[1]
    grid_points = np.zeros((xx.size, n_features))
    grid_points[:, i] = xx.ravel()
    grid_points[:, j] = yy.ravel()

    # Fill other features with mean values
    for k in range(n_features):
        if k not in feature_indices:
            grid_points[:, k] = X[:, k].mean()

    # Predict
    try:
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='k', linewidths=0.5)
    except Exception as e:
        warnings.warn(f"Could not compute decision boundary: {e}")

    # Plot data points
    scatter = ax.scatter(X[:, i], X[:, j], c=y, cmap='RdYlBu',
                         edgecolors='k', s=50)

    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel(f'Feature {j}')
    ax.set_title('Decision Boundary')
    plt.colorbar(scatter, ax=ax, label='Class')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_barren_plateau_analysis(
    analysis_results: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot barren plateau analysis results.

    Args:
        analysis_results: Dictionary from barren plateau analysis
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Cost landscape (if available)
    ax1 = axes[0]
    if 'theta_samples' in analysis_results and 'cost_samples' in analysis_results:
        thetas = analysis_results['theta_samples']
        costs = analysis_results['cost_samples']
        ax1.plot(thetas, costs, 'b-', linewidth=2)

        if 'optimal_theta' in analysis_results:
            opt_theta = analysis_results['optimal_theta']
            opt_cost = analysis_results['optimal_cost']
            ax1.axvline(opt_theta, color='r', linestyle='--', label='Optimal')
            ax1.scatter([opt_theta], [opt_cost], color='r', s=100, zorder=5)

    ax1.set_xlabel('Parameter θ')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost Landscape')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Fourier coefficients
    ax2 = axes[1]
    if 'a_coeffs' in analysis_results and 'b_coeffs' in analysis_results:
        a_coeffs = analysis_results['a_coeffs']
        b_coeffs = analysis_results['b_coeffs']
        n_coeffs = len(a_coeffs)
        x = np.arange(n_coeffs)

        ax2.bar(x - 0.2, a_coeffs, 0.4, label='cos coefficients', color='blue')
        ax2.bar(x + 0.2, b_coeffs, 0.4, label='sin coefficients', color='orange')

    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Coefficient')
    ax2.set_title('Fourier Coefficients')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Metrics summary
    ax3 = axes[2]
    ax3.axis('off')

    metrics_text = "Analysis Metrics:\n\n"

    key_metrics = [
        ('amplitude', 'Amplitude'),
        ('curvature', 'Curvature'),
        ('cost_variance', 'Cost Variance'),
        ('cost_range', 'Cost Range'),
        ('is_flat', 'Is Flat'),
        ('barren_plateau_detected', 'Barren Plateau')
    ]

    for key, label in key_metrics:
        if key in analysis_results:
            value = analysis_results[key]
            if isinstance(value, bool):
                value = 'Yes' if value else 'No'
            elif isinstance(value, float):
                value = f'{value:.6f}'
            metrics_text += f"{label}: {value}\n"

    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.set_title('Summary')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison_results(
    results: Dict[str, Dict],
    metrics: List[str] = ['accuracy', 'training_time', 'n_gates'],
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models.

    Args:
        results: Dictionary mapping model names to their results
        metrics: Metrics to compare
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    model_names = list(results.keys())
    x = np.arange(len(model_names))

    for ax, metric in zip(axes, metrics):
        values = [results[name].get(metric, 0) for name in model_names]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
        bars = ax.bar(x, values, color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}' if isinstance(val, float) else str(val),
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_circuit_growth(
    history: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot circuit growth during adaptive construction.

    Args:
        history: Training history with gate information
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    iterations = range(len(history))

    # Number of gates
    n_gates = [h.get('n_gates', 0) for h in history]
    ax1.plot(iterations, n_gates, 'b-o', linewidth=2, markersize=4)
    ax1.fill_between(iterations, n_gates, alpha=0.3)
    ax1.set_ylabel('Number of Gates')
    ax1.set_title('Adaptive Circuit Growth')
    ax1.grid(True, alpha=0.3)

    # Cost
    costs = [h.get('cost', 0) for h in history]
    ax2.plot(iterations, costs, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
