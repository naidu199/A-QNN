# ARC-QNN: Analytic Iterative Circuit Reconstruction for Quantum Neural Networks

A **gradient-free** approach to training Quantum Neural Networks that **mitigates the barren plateau problem** by constructing circuits gate-by-gate using analytic parameter estimation instead of gradient descent.

## Problem Statement

Traditional Variational Quantum Classifiers (VQCs) suffer from the **barren plateau problem**:

- Optimization landscape becomes exponentially flat as qubits increase
- Gradients vanish, making gradient-based training (COBYLA, SPSA, Adam) ineffective
- Fixed ansatz circuits waste parameters on uninformative gates

## Our Solution: ARC (Analytic Iterative Circuit Reconstruction)

ARC takes a fundamentally different approach — no gradients, no fixed ansatz, no barren plateaus:

1. **Empty Start**: Circuit begins with zero gates (no random initialization)
2. **Gate Pool**: 6 gate types × n_qubits positions = candidate gates per iteration
3. **3-Point Rotosolve**: Evaluate each candidate at θ = 0, π/2, −π/2 to analytically solve `P(θ) = a·cos(θ·x − b) + c`
4. **Greedy Selection**: Pick the gate + feature + θ that minimizes log-loss across all candidates
5. **Patience-Based Convergence**: Stop when no improvement for `patience` consecutive gates, revert to best circuit found

### ARC vs Fixed Ansatz VQC

| Feature                | Fixed Ansatz VQC          | ARC QNN                    |
| ---------------------- | ------------------------- | -------------------------- |
| Circuit Structure      | Pre-defined layers        | Built gate-by-gate         |
| Parameter Optimization | Gradient descent (COBYLA) | 3-point analytic solve     |
| Initialization         | Random parameters         | Empty circuit              |
| Barren Plateaus        | Severely affected         | Mitigated by design        |
| Gate Count             | Fixed (e.g. 60 params)    | Adaptive (e.g. 7-14 gates) |
| MNIST 0v1 Accuracy     | 44.8%                     | **92.9%**                  |

### Gate Types

| Gate ID | Operation      | Description                         |
| ------- | -------------- | ----------------------------------- |
| U1      | Rz(θ·x)        | Z-rotation parameterized by feature |
| U2      | Rx(θ·x)        | X-rotation parameterized by feature |
| U3      | Ry(θ·x)        | Y-rotation parameterized by feature |
| H       | H + Rz(θ·x)    | Hadamard followed by Z-rotation     |
| X       | CNOT + Rz(θ·x) | Entangling gate + Z-rotation        |
| Z       | CZ + Rz(θ·x)   | Controlled-Z + Z-rotation           |

## Project Structure

```
A-QNN/
├── compare_qnn.py              # Main CLI entry point
├── train.py                    # Alternative training script
├── requirements.txt            # Dependencies
│
├── src/
│   ├── estimators/
│   │   ├── arc_estimator.py        # Core ARC algorithm (ARCEstimator, ARCGatePool)
│   │   ├── analytic_estimator.py   # Analytic parameter estimation
│   │   ├── fourier_estimator.py    # Fourier-based methods
│   │   └── gradient_free.py        # Gradient-free optimizers
│   │
│   ├── models/
│   │   ├── fixed_ansatz_qnn.py     # Fixed VQC baseline (H+RY encoding, RY+RZ+CNOT layers)
│   │   ├── adaptive_qnn.py        # Adaptive QNN model
│   │   └── qnn_classifier.py      # Sklearn-compatible wrapper
│   │
│   ├── evaluation/
│   │   ├── comparison.py          # ARC vs Fixed comparison pipeline
│   │   ├── ibm_runner.py          # IBM Quantum backend runner
│   │   ├── metrics.py             # Performance metrics
│   │   └── visualization.py       # Plotting utilities
│   │
│   ├── circuits/
│   │   ├── quantum_gates.py       # Gate operations
│   │   ├── circuit_builder.py     # Circuit builder
│   │   └── encoding.py            # Data encoding strategies
│   │
│   ├── data/
│   │   ├── preprocessing.py       # MinMax scaling, PCA
│   │   └── datasets.py            # Dataset loaders
│   │
│   ├── training/
│   │   ├── trainer.py             # Training orchestration
│   │   └── callbacks.py           # Training callbacks
│   │
│   └── utils/
│       ├── helpers.py             # Helper functions
│       ├── config.py              # Configuration
│       └── hyperparameter_tuning.py
│
├── Datasets/                   # Pre-generated datasets (CSV)
│   ├── Xmnist_train_0_1_7x7_N_1000.csv
│   ├── Xmnist-pca_train_10d_11552.csv
│   ├── Xtwo-curves_train_1000.csv
│   └── ...
│
├── docs/
│   ├── TRAINING_EXPLAINED.md      # Detailed ARC training walkthrough
│   ├── architecture_diagram.drawio
│   ├── 01_introduction.md
│   └── abstract.md
│
├── results/                    # Comparison outputs
│   ├── mnist_0_1_7x7/
│   ├── mnist_pca_10d/
│   └── two-curves/
│
└── Analytic-QNN-Reconstruction/  # Reference implementation (Qiskit 0.45)
```

## Usage

### Run ARC vs Fixed Ansatz Comparison

```bash
# Activate virtual environment
.\venv\Scripts\activate

# MNIST 0 vs 1 (7x7 downsampled, 10 qubits)
python compare_qnn.py --dataset mnist_0_1_7x7 --n_qubits 10 --methods arc cobyla --patience 2

# MNIST PCA (10 features, auto 11 qubits)
python compare_qnn.py --dataset mnist_pca_10d --methods arc cobyla --patience 2

# Two-curves (4 features, auto 5 qubits)
python compare_qnn.py --dataset two-curves --methods arc cobyla

# Auto mode (auto-applies PCA if features > max_qubits)
python compare_qnn.py --dataset mnist_0_1_28x28 --max_qubits 15 --methods arc cobyla
```

### CLI Options

| Flag               | Default      | Description                                 |
| ------------------ | ------------ | ------------------------------------------- |
| `--dataset`        | `two-curves` | Dataset name (matches files in `Datasets/`) |
| `--n_qubits`       | `0` (auto)   | Number of qubits (0 = n_features + 1)       |
| `--max_qubits`     | `20`         | Auto-apply PCA if features exceed this      |
| `--methods`        | `arc cobyla` | Methods to compare                          |
| `--arc_max_gates`  | `15`         | Maximum gates ARC will try                  |
| `--patience`       | `1`          | Gates without improvement before stopping   |
| `--fixed_layers`   | `3`          | Layers in fixed ansatz VQC                  |
| `--fixed_max_iter` | `100`        | COBYLA optimizer iterations                 |

### Training Output

Each ARC gate iteration prints:

```
Gate 1 | Cost: 0.6812 | Type: U3(Ry) | Qubit: 0 | Feature: 3 | θ: 0.4521 | Time: 0.8s
Gate 2 | Cost: 0.5234 | Type: U2(Rx) | Qubit: 0 | Feature: 7 | θ: -0.3112 | Time: 0.7s
...
Gate 7 | Cost: 0.2891 | Type: U3(Ry) | Qubit: 0 | Feature: 1 | θ: 0.6743 | Time: 0.9s
Converged at gate 7 (patience=1)
```

## Datasets

Pre-generated CSV datasets in `Datasets/`:

| Dataset               | Features                | Train Size | Test Size |
| --------------------- | ----------------------- | ---------- | --------- |
| `mnist_0_1_7x7`       | 49 (needs PCA/n_qubits) | 1,000      | 1,000     |
| `mnist_0_1_14x14`     | 196                     | 1,000      | 1,000     |
| `mnist_3_5_7x7`       | 49                      | 1,000      | 1,000     |
| `mnist-pca_10d`       | 10                      | 11,552     | 1,902     |
| `two-curves`          | 4                       | 1,000      | 1,000     |
| `linearly-separable`  | 4                       | 1,000      | 1,000     |
| `bars-and-stripes_4d` | 4                       | 1,000      | —         |

## Comparison Results

### MNIST 0 vs 1 (7x7, 10 qubits, patience=2)

| Metric         | ARC QNN    | Fixed COBYLA |
| -------------- | ---------- | ------------ |
| Test Accuracy  | **92.90%** | 44.80%       |
| Gates / Params | 14         | 60           |
| Circuit Depth  | 14         | 38           |
| Training Time  | ~109s      | ~72s         |

### MNIST PCA 10D (11 qubits, patience=2)

| Metric        | ARC QNN    | Fixed COBYLA |
| ------------- | ---------- | ------------ |
| Test Accuracy | **75.50%** | 53.10%       |
| Gates         | 2          | 66           |
| Training Time | ~30s       | ~48s         |

### Two-Curves (5 qubits, patience=1)

| Metric        | ARC QNN    | Fixed COBYLA |
| ------------- | ---------- | ------------ |
| Test Accuracy | **57.50%** | 50.70%       |
| Gates         | 5          | 30           |
| Training Time | ~13s       | ~8s          |

## Speed Optimizations

ARC training was optimized from **255s/gate → 0.8s/gate** (~300× speedup):

| Optimization            | Speedup | Description                                     |
| ----------------------- | ------- | ----------------------------------------------- |
| ABC feature-independent | ~11×    | Compute (a,b,c) once, reuse across all features |
| Vectorized NumPy        | ~5×     | Broadcasting replaces Python loops              |
| Pre-bound base circuits | ~3×     | Reuse `assign_parameters()` across gate pool    |
| Pre-built test gates    | ~2×     | Build 3 test circuits once per candidate        |
| Joblib parallelism      | ~2×     | Per-sample parallelism across CPU cores         |
| Subsampling (100/iter)  | ~10×    | Random subset per iteration                     |

## How It Works

1. **Data Preprocessing**: Features scaled to [0, π], bias column (π) appended, `n_qubits = n_features + 1`
2. **ARC Loop**: For each gate iteration:
   - Generate all candidate gates (6 types × n_qubits)
   - For each candidate, evaluate circuit at θ = 0, π/2, −π/2 → get P(|0...0⟩) probabilities
   - Fit sinusoidal model: `P(θ) = a·cos(θ·x_f − b) + c` for each sample
   - Grid-search over θ ∈ [−1, 1] to minimize log-loss for each feature
   - Select the (gate, feature, θ) triple with lowest cost
3. **Prediction**: Run final circuit with input features → measure P(|0...0⟩) → classify (P > 0.5 → Class 0)

See [docs/TRAINING_EXPLAINED.md](docs/TRAINING_EXPLAINED.md) for a detailed walkthrough with formulas and examples.

## References

- Analytic Iterative Circuit Reconstruction — original reference implementation in `Analytic-QNN-Reconstruction/`
- Rotosolve algorithm for analytic parameter optimization in quantum circuits
- Ostaszewski et al., "Structure optimization for parameterized quantum circuits"

## Requirements

- Python 3.9+
- Qiskit 1.x with qiskit-aer
- NumPy, SciPy, scikit-learn
- joblib (parallelism)
- matplotlib (visualization)

```bash
pip install -r requirements.txt
```

## License

MIT License — See LICENSE file for details.
