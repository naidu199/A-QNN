# Chapter 4: Methodology and Design

This chapter describes the methodology and design of the Adaptive Quantum Neural Network (A-QNN) system. It covers the overall proposed model architecture, the feature engineering pipeline that prepares classical data for quantum circuits, the loss function and optimizer configuration used during training, and the evaluation metrics used to assess model performance. Each section is accompanied by a draw.io diagram stored in the `docs/` folder; open the `.drawio` file in [app.diagrams.net](https://app.diagrams.net) to view or edit the diagram.

---

## 4.1 Proposed Model Architecture

The A-QNN system is organized into four major stages that execute sequentially during a full training and inference run: (1) Data Ingestion and Preprocessing, (2) ARC Circuit Training via adaptive gate construction and analytic Rotosolve, (3) Circuit Evaluation where the trained circuit is applied to test data, and (4) Performance Assessment.

The `AdaptiveQNN` class (`src/models/adaptive_qnn.py`) acts as the top-level controller, coordinating the `ARCEstimator` for circuit construction, the `DataEncoder` for feature embedding, and the `AerSimulator` backend for statevector evaluation. The `FixedAnsatzQNN` class (`src/models/fixed_ansatz_qnn.py`) provides the baseline VQC that is trained in parallel via COBYLA for comparison. Both models share the same preprocessing pipeline and evaluation infrastructure, ensuring a fair controlled comparison.

The ARC core loop inside `ARCEstimator.reconstruct_circuit()` maintains an incrementally growing `QuantumCircuit`. At each iteration it:

1. Samples a training subset
2. Pre-binds the base circuit to each training sample
3. Evaluates all gate candidates at three test angles
4. Computes $(a, b, c)$ coefficients per sample via Rotosolve
5. Finds the globally optimal $\theta^*$ via vectorized grid search
6. Appends the best gate permanently and checks the patience counter

**Diagram 4.1 — Overall System Architecture** → [diagram_4_1_system_architecture.drawio](diagram_4_1_system_architecture.drawio)

---

## 4.2 Feature Engineering Pipeline

Raw datasets arrive as CSV files with feature dimensionalities ranging from 4 (linearly separable) to 785 (MNIST 28×28). The feature engineering pipeline transforms these into a fixed-width, properly scaled quantum-ready matrix before any circuit operations take place.

The pipeline runs in the following order, implemented across `compare_qnn.py` and `src/data/preprocessing.py`:

1. **Load CSV** — `numpy.loadtxt` reads `X` and `Y` files separately.
2. **Label normalisation** — Maps $\{-1, +1\}$ labels to $\{0, 1\}$.
3. **PCA** (conditional) — Applied only when raw features $d > n_{\text{qubits}}$; fitted on train, transformed on both train and test to prevent leakage.
4. **Min-Max Scaling** — Maps all features to $[0, \pi]$ using `MinMaxScaler` fitted on train only.
5. **Bias Append** — Concatenates a constant column $x_d = \pi$ for all samples.
6. **Output** — Feature matrix of shape $(N, n_{\text{features}} + 1)$ ready for ARC training.

**Diagram 4.2 — Feature Engineering Pipeline** → [diagram_4_2_feature_engineering.drawio](diagram_4_2_feature_engineering.drawio)

---

## 4.3 Loss Function, Optimizer, and Hyperparameter Configuration

### 4.3.1 Loss Function

The ARC algorithm minimizes the **binary cross-entropy (log-loss)** at every gate selection step:

$$C(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i(\theta) + (1 - y_i) \log(1 - \hat{p}_i(\theta)) \right]$$

where $\hat{p}_i(\theta) = P(|0\cdots0\rangle \mid \mathbf{x}_i, \theta)$ is computed via statevector simulation. The implementation uses a vectorized NumPy computation (replacing 1,000 individual `sklearn.log_loss` calls):

```python
costs = -(y[None,:] * np.log(expect) + (1 - y[None,:]) * np.log(1 - expect)).mean(axis=1)
```

The fixed-ansatz COBYLA baseline also uses log-loss as its cost function, ensuring both methods optimize the same objective for a fair comparison.

### 4.3.2 ARC Optimizer — Analytic Rotosolve

ARC does not use a classical iterative optimizer. Instead, for each gate candidate the optimal parameter is found **analytically** in a single pass:

| Step | Operation                                                                 | Cost                |
| ---- | ------------------------------------------------------------------------- | ------------------- |
| 1    | Evaluate circuit at $\theta = 0, +\pi/2, -\pi/2$                          | 3 statevector calls |
| 2    | Solve $(a_i, b_i, c_i)$ per sample                                        | Closed form         |
| 3    | Reconstruct $C(\theta)$ landscape                                         | Vectorized NumPy    |
| 4    | Grid search $\theta^* = \arg\min C(\theta)$ over 1001 points in $[-1, 1]$ | $O(1001 \times N)$  |

This eliminates learning rate tuning, gradient estimation overhead, and convergence sensitivity.

### 4.3.3 Fixed-Ansatz Optimizer — COBYLA

The baseline VQC is optimized with **COBYLA** (Constrained Optimization BY Linear Approximations, Powell 1994) via `scipy.optimize.minimize`. COBYLA builds a linear approximation of the objective from direct function evaluations without computing derivatives, making it suitable for quantum circuits where gradients require additional circuit evaluations. The circuit parameters (one $R_Y$ and one $R_Z$ per qubit per layer) are all optimized simultaneously.

### 4.3.4 Hyperparameter Configuration

The table below documents all hyperparameters and their values used in the primary experiment (MNIST 0 vs 1, 7×7, 10 qubits):

| Hyperparameter           | ARC Value           | Fixed Ansatz Value | Description                          |
| ------------------------ | ------------------- | ------------------ | ------------------------------------ |
| `n_qubits`               | 10                  | 10                 | Number of qubits                     |
| `max_gates` / `n_layers` | 15                  | 3                  | Max gates (ARC) / layers (Fixed)     |
| `gate_pool`              | U1, U2, U3, H, X, Z | —                  | Gate types in ARC pool               |
| `subsample_size`         | 100                 | 200                | Training samples per cost evaluation |
| `patience`               | 3                   | —                  | Non-improving gates before stopping  |
| `max_iter`               | —                   | 100                | COBYLA max iterations                |
| `n_jobs`                 | −1 (all cores)      | 1                  | Parallel CPU cores                   |
| `seed`                   | 42                  | 42                 | Random seed for reproducibility      |
| `optimizer`              | Analytic Rotosolve  | COBYLA             | Optimization method                  |
| `cost_function`          | Binary log-loss     | Binary log-loss    | Training objective                   |
| `theta_grid_size`        | 1001                | —                  | Grid points for θ\* search           |
| `theta_grid_range`       | $[-1, 1]$           | —                  | Search range for θ\*                 |

**Diagram 4.3 — ARC Training Loop Detail** → [diagram_4_3_arc_training_loop.drawio](diagram_4_3_arc_training_loop.drawio)

---

## 4.4 Evaluation Metrics

Model performance is assessed using a comprehensive set of classification metrics implemented in `src/evaluation/metrics.py` and `src/evaluation/comparison.py`. Metrics are computed independently for the training set (to detect overfitting) and the held-out test set (to measure generalization).

### 4.4.1 Primary Metrics

| Metric                | Formula                                                                                 | What It Measures                                      |
| --------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Accuracy**          | $\frac{TP + TN}{N}$                                                                     | Overall fraction of correct predictions               |
| **Balanced Accuracy** | $\frac{1}{2}\!\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$                         | Average per-class accuracy; robust to class imbalance |
| **F1 Score**          | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall                 |
| **Precision**         | $\frac{TP}{TP + FP}$                                                                    | Fraction of positive predictions that are correct     |
| **Recall**            | $\frac{TP}{TP + FN}$                                                                    | Fraction of true positives correctly identified       |
| **Log-Loss**          | $-\frac{1}{N}\sum y_i \log \hat{p}_i$                                                   | Quality of probability calibration                    |
| **AUC-ROC**           | Area under ROC curve                                                                    | Ranking quality across all thresholds                 |

### 4.4.2 Circuit-Level Metrics

Beyond classification performance, the following circuit-level metrics are recorded to assess the structural efficiency of each method:

| Metric                 | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| **Circuit Depth**      | Number of sequential gate layers in the final circuit             |
| **Num Parameters**     | Total trainable rotation angles                                   |
| **Num Gates**          | Total gate count (ARC: equals parameters; Fixed: can differ)      |
| **Training Time (s)**  | Wall-clock seconds from first gate evaluation to convergence      |
| **Total Measurements** | Total number of statevector evaluations performed during training |

### 4.4.3 Comparison Protocol

The evaluation follows a fixed protocol to ensure fairness:

1. Both models receive **identical preprocessed training and test splits**.
2. All results are recorded at identical **random seed 42**.
3. Test metrics are computed on the **held-out test set only** (never used during training or hyperparameter tuning).
4. Final comparison is tabulated in `results/<dataset>/comparison_report.txt` and serialized to `comparison_results.json` for further analysis.
5. A **hardware simulation pass** additionally evaluates both trained circuits using `AerSimulator` with 4096 shots to estimate shot-noise sensitivity of the final predictions.

---

_References for this chapter are consolidated in the project bibliography._
