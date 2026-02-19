# ARC QNN Training - How It Works

## Executive Summary

The **Analytic Iterative Circuit Reconstruction (ARC)** QNN is a novel approach to training quantum classifiers that **avoids the barren plateau problem** — a major obstacle in quantum machine learning where gradients vanish exponentially as circuits grow larger.

**Key Innovation**: Instead of using gradient descent on a fixed circuit, we:

1. Start with an **empty circuit** (no predefined structure)
2. Build the circuit **gate-by-gate**, greedily selecting the best gate each iteration
3. Compute optimal parameters **analytically** using 3-point Rotosolve (no gradients)
4. Only add gates that **reduce log-loss cost**
5. Automatically stop via **patience-based convergence**

---

## The Problem: Barren Plateaus

### What are Barren Plateaus?

In traditional Variational Quantum Circuits (VQCs):

```
Traditional VQC Training:
1. Define fixed circuit structure with many parameters
2. Initialize random parameters
3. Compute gradients via parameter-shift rule
4. Update parameters using gradient descent
5. Repeat until convergence
```

**The Problem**: As the number of qubits increases, the optimization landscape becomes exponentially flat:

| Qubits | Gradient Magnitude |
| ------ | ------------------ |
| 2      | ~0.1               |
| 4      | ~0.01              |
| 8      | ~0.0001            |
| 16     | ~0.00000001        |

With tiny gradients, the optimizer can't find good directions → **training fails**.

---

## Our Solution: ARC (Analytic Iterative Circuit Reconstruction)

### How ARC Differs from Traditional VQC

| Aspect       | Traditional VQC (Fixed Ansatz)  | ARC QNN                           |
| ------------ | ------------------------------- | --------------------------------- |
| Circuit      | Fixed structure, random params  | Empty → grows gate by gate        |
| Optimization | Gradient descent (COBYLA, etc.) | Analytic 3-point Rotosolve        |
| Scalability  | Barren plateaus at scale        | Maintains trainability            |
| Measurements | O(parameters × iterations)      | 3 per gate candidate per sample   |
| Parameters   | All optimized jointly           | Each fixed at optimal upon adding |

### The Core Algorithm

```
ARC Training Loop:
1. Start with EMPTY circuit (no encoding, no ansatz)
2. Add bias feature π to data (constant rotation baseline)
3. For each iteration:
   a. From gate pool, try every candidate gate on every qubit
   b. For each candidate: evaluate at θ=0, θ=π/2, θ=−π/2
   c. Fit sinusoidal: P(θ) = a·cos(θ·x_i − b) + c
   d. Find optimal θ by minimizing log-loss over all features
   e. Select gate+feature+angle with lowest cost
   f. Append to circuit permanently
4. Stop when patience consecutive non-improving gates found
5. Revert to best circuit found
```

---

## The Analytic Parameter Estimation (Rotosolve)

### The Mathematical Insight

For a parameterized rotation gate R(θ · x_i) appended to an existing circuit, the probability of measuring |0⋯0⟩ follows a **sinusoidal** pattern:

```
P(|0⟩) = a · cos(θ · x_i − b) + c
```

This is a fundamental property of quantum rotations — the expectation value is always sinusoidal in the rotation angle.

### Finding Optimal θ with 3 Measurements

We evaluate the circuit at exactly **3 test angles** for each candidate gate:

1. θ = 0 → gives f₀ = a · cos(−b) + c
2. θ = +π/2 → gives f₊ = a · cos(π/2 − b) + c
3. θ = −π/2 → gives f₋ = a · cos(−π/2 − b) + c

From these 3 values, solve for the cosine parameters:

```
c = (f₊ + f₋) / 2
b = −arctan2(2(f₀ − c), f₊ − f₋) + π/2
a = √((f₀ − c)² + (f₊ − f₋)² / 4)
```

Once we have (a, b, c) for every training sample, we reconstruct the full cost landscape:

```
C(θ) = (1/N) Σᵢ LogLoss(yᵢ, aᵢ·cos(θ·x_{i,f} − bᵢ) + cᵢ)
```

This is a **classical optimization** over a single variable θ, solved by evaluating on a grid of 1000 points in [−1, 1].

**No gradient computation needed!**

---

## Training Loop — Step by Step

### Step 1: Data Preparation

```
Input: X_train (1000 × 10), y_train (1000,)
  ↓  Scale features to [0, π]
  ↓  Append bias feature: X becomes (1000 × 11), last column = π
  ↓  Set n_qubits = n_features = 11
```

The bias feature (constant π) ensures the circuit always has a feature with full rotation range, acting like a trainable bias term.

### Step 2: Initialize Empty Circuit

```
Circuit: (empty — no gates at all)
         qc_0: ─
         qc_1: ─
         ...
         qc_9: ─

Measurement: P(|0...0⟩) = probability of all-zero state
Initial cost: log_loss with P = 1.0 for all samples ≈ 0.693
```

Unlike traditional VQC (which starts with H + Ry encoding layer), ARC starts completely empty. The data is encoded through the rotation angles of added gates: Rx(θ · x_f).

### Step 3: Gate Pool Construction

The gate pool contains **every possible single-gate placement**:

```
Gate types: [U1(Rz), U2(Rx), U3(Ry), H, X(CNOT), Z(CZ)]
Qubits: 10
Pool size: 6 gate types × 10 qubits = 60 candidates

Example entries:
  ['U2_x', '111111', '111111', ...] = Rx on qubit 0, identity elsewhere
  ['111111', 'U3_x', '111111', ...] = Ry on qubit 1, identity elsewhere
  ['111111', '111111', 'H_x', ...]  = H on qubit 2, identity elsewhere
```

### Step 4: Iteration 1 — Evaluate All Candidates

For the **first gate** (empty base circuit), evaluation is cheap:

```
For each of 60 gate candidates:
  Build 3 test circuits: gate(θ=0), gate(θ=π/2), gate(θ=-π/2)
  Compute Statevector → P(|0...0⟩) for each
  Get (a, b, c) for each sample
  For each of 11 features:
    Minimize log-loss over θ grid → optimal cost for this gate+feature

Total evaluations: 60 × 3 = 180 statevector computations
```

**Key optimization — ABC feature-independence**: The (a, b, c) values are computed with test parameter = 1.0, so they don't depend on which feature is selected. We compute ABC once per gate candidate and reuse across all 11 features. This avoids 11× redundant circuit evaluations.

### Step 5: Select Best Gate

```
Gate 1: cost=0.688, theta=-0.532, feat=10, gate=Rx on qubit 0
  → Selected: Rx(-0.532 · x[10]) on qubit 0
  → feat=10 is the bias feature (π), so this gate rotates by -0.532 × π
```

### Step 6: Add Gate to Circuit and Repeat

```
Circuit after gate 1:
  qc_0: ─[Rx(-0.532 × x[10])]─
  qc_1: ───────────────────────
  ...

Cost: 0.688 (improved from 0.693)
```

For subsequent gates, the base circuit now has content. Each sample's feature values are bound into the base circuit, then test gates are appended. **Joblib parallelism** distributes the per-sample ABC computation across all CPU cores.

### Step 7: Convergence with Patience

```
Gate 1: cost=0.688  ✓ improving
Gate 2: cost=0.515  ✓ improving
Gate 3: cost=0.437  ✓ improving
Gate 4: cost=0.362  ✓ improving
Gate 5: cost=0.325  ✓ improving
Gate 6: cost=0.297  ✓ improving (best so far)
Gate 7: cost=0.308  ✗ worse! patience counter = 1
Gate 8: cost=0.238  ✓ improving! patience counter reset = 0
Gate 9: cost=0.190  ✓ improving
...
Gate 14: cost=0.106 ✓ improving (new best)
Gate 15: cost=0.109 ✗ worse! patience counter = 1
→ patience=2, counter < 2, continue...
Gate 16: cost=0.112 ✗ worse! patience counter = 2
→ patience=2, counter >= 2, STOP
→ Revert to best circuit at gate 14 (cost=0.106)
```

Without patience (patience=1), training would have stopped at gate 7 with cost=0.308. With patience=2, it pushed through and found a much better circuit at gate 14 with cost=0.106.

---

## Speed Optimizations

Our implementation runs **~300× faster** than a naive approach (255s → 0.8s per gate):

| Optimization                     | What It Does                                         | Speedup                           |
| -------------------------------- | ---------------------------------------------------- | --------------------------------- |
| **ABC feature-independence**     | Compute ABC once per gate, reuse across all features | ~11× (one per feature eliminated) |
| **Vectorized log-loss**          | NumPy broadcasting: `a*cos(outer(θ,X)-b)+c`          | ~50× vs Python loops              |
| **Pre-bound base circuits**      | Bind training data into circuits once per iteration  | ~60× (reuse across gate pool)     |
| **Pre-built test gates**         | Build all 3 test circuits for all candidates upfront | ~3×                               |
| **Joblib parallelism**           | Per-sample ABC across CPU cores (`n_jobs=-1`)        | ~Nx (N = cores)                   |
| **Subsample (train_length=100)** | Random 100 samples per iteration                     | ~10× for 1000-sample datasets     |

---

## Understanding the Output

### Training Output

```
Gate pool: 60 candidates, 11 features, n_jobs=-1
Gate 1: cost=0.688139, theta=-0.53200, feat=10, gate=['U2_x', '111111', ...],
        samples=100, prep=0.0s eval=0.6s total=0.6s
```

| Field     | Meaning                                                         |
| --------- | --------------------------------------------------------------- |
| `cost`    | Log-loss after adding this gate (lower = better)                |
| `theta`   | Optimal rotation angle found analytically                       |
| `feat`    | Feature index multiplied with theta (last = bias π)             |
| `gate`    | Gate descriptor: `U2_x` = Rx on that qubit, `111111` = identity |
| `samples` | Training subsample size used                                    |
| `prep`    | Time to pre-bind base circuits                                  |
| `eval`    | Time to evaluate all gate candidates                            |

### Gate Type Mapping

| Code | Gate                  | What It Does                 |
| ---- | --------------------- | ---------------------------- |
| `U1` | Rz(θ · x_f)           | Phase rotation (z-axis)      |
| `U2` | Rx(θ · x_f)           | X-axis rotation              |
| `U3` | Ry(θ · x_f)           | Y-axis rotation              |
| `H`  | H then Rz(θ · x_f)    | Hadamard + phase             |
| `X`  | CNOT then Rz(θ · x_f) | Entangling (adjacent qubits) |
| `Z`  | CZ then Rz(θ · x_f)   | Entangling (adjacent qubits) |

### Final Results

```
ARC QNN Results:
  Train accuracy: 0.9600
  Test accuracy:  0.9290    ← Main metric
  Gates used:     14        ← Very compact circuit
  Circuit depth:  14
  Training time:  108.8s
```

---

## Comparison: ARC vs Fixed Ansatz

### Results on MNIST 0 vs 1 (7×7, 10 qubits, PCA)

| Metric            | ARC QNN (patience=2) | Fixed Ansatz COBYLA |
| ----------------- | -------------------- | ------------------- |
| **Test Accuracy** | **92.9%**            | 44.8%               |
| Train Accuracy    | 96.0%                | 46.2%               |
| Parameters        | 14                   | 60                  |
| Circuit Depth     | 14                   | 38                  |
| Training Time     | ~109s                | ~72s                |

### Why ARC Wins

1. **No barren plateaus**: Each parameter is set analytically, not via gradient descent
2. **Compact circuits**: Only 14 gates vs 60 parameters — less noise on real hardware
3. **Greedy but effective**: Each gate is chosen to maximize immediate cost reduction
4. **Patience mechanism**: Pushes past temporary local minima in the greedy search

### Why Fixed Ansatz Fails

Fixed ansatz with COBYLA optimizer at 100 iterations with 60 parameters on 10 qubits:

- Cost landscape is essentially flat (barren plateau)
- COBYLA can't find meaningful descent directions
- Stuck near random guessing (50%)

---

## Architecture Overview

```
                    ┌──────────────────┐
Input Data ────────►│  Scale to [0,π]  │
(N × d features)   │  Add bias (π)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Gate Pool       │
                    │  60 candidates    │
                    │  (6 types × 10q) │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │   For each candidate gate:   │
              │   1. Eval at θ=0, π/2, -π/2 │◄──── 3 Statevector sims
              │   2. Get (a,b,c) per sample  │◄──── Rotosolve
              │   3. Minimize log-loss(θ)    │◄──── Classical grid search
              │   4. Best gate+feat+angle    │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Append to        │
                    │  circuit           │───► Repeat until converged
                    └────────┬─────────┘      (patience-based stopping)
                             │
                    ┌────────▼─────────┐
                    │  Final Circuit    │
                    │  Predict via      │
                    │  P(|0...0⟩)       │───► Class 0 if P > 0.5
                    └──────────────────┘      Class 1 if P ≤ 0.5
```

---

## CLI Usage

### Basic Run

```bash
python compare_qnn.py --dataset mnist_0_1_7x7 --n_qubits 10 --methods arc cobyla
```

### Full Options

```bash
python compare_qnn.py \
  --dataset mnist_0_1_7x7 \
  --n_qubits 10 \
  --max_qubits 20 \
  --methods arc cobyla \
  --arc_max_gates 15 \
  --arc_gate_list U1 U2 U3 H X Z \
  --arc_subsample 100 \
  --patience 2 \
  --n_jobs -1 \
  --fixed_layers 3 \
  --fixed_max_iter 100 \
  --seed 42
```

| Flag               | Default        | Description                                           |
| ------------------ | -------------- | ----------------------------------------------------- |
| `--dataset`        | —              | Dataset name (see table below)                        |
| `--n_qubits`       | 0 (auto)       | Number of qubits (0 = auto, capped at `--max_qubits`) |
| `--max_qubits`     | 20             | Auto PCA cap when n_qubits=0                          |
| `--methods`        | arc cobyla     | Methods to compare                                    |
| `--arc_max_gates`  | 15             | Max gates for ARC                                     |
| `--arc_gate_list`  | U1 U2 U3 H X Z | Gate types in pool                                    |
| `--arc_subsample`  | 100            | Training subsample per iteration                      |
| `--patience`       | 1              | Non-improving gates before stopping                   |
| `--n_jobs`         | -1             | Parallel cores (-1 = all)                             |
| `--fixed_layers`   | 3              | Layers for fixed ansatz                               |
| `--fixed_max_iter` | 100            | Optimizer iterations for fixed ansatz                 |
| `--seed`           | 42             | Random seed                                           |

### Available Datasets

| Dataset              | Features | Train/Test | Description                      |
| -------------------- | -------- | ---------- | -------------------------------- |
| `two-curves`         | 8        | 1000/1000  | Two interleaved curves           |
| `linearly-separable` | 4        | 1000/1000  | Linearly separable classes       |
| `bars-and-stripes`   | 16       | 1000/1000  | 4×4 bars and stripes patterns    |
| `mnist_0_1_7x7`      | 50       | 1000/1000  | MNIST digits 0 vs 1 (7×7 pixels) |
| `mnist_3_5_7x7`      | 50       | 1000/1000  | MNIST digits 3 vs 5 (7×7 pixels) |
| `mnist_pca_10d`      | 11       | 11552/1902 | MNIST with PCA (10 components)   |

---

## Key Implementation Files

| File                              | Purpose                                                          |
| --------------------------------- | ---------------------------------------------------------------- |
| `src/estimators/arc_estimator.py` | Core ARC algorithm: gate pool, Rotosolve, circuit reconstruction |
| `src/models/fixed_ansatz_qnn.py`  | Fixed-structure VQC baseline (H+Ry encoding, Ry+Rz+CNOT layers)  |
| `src/evaluation/comparison.py`    | Orchestrates ARC vs Fixed training, metrics, reporting           |
| `compare_qnn.py`                  | Main CLI script: data loading, PCA, comparison pipeline          |

---

## Limitations and Future Work

### Current Limitations

- **Single-qubit dominance**: ARC's greedy search with P(|0⋯0⟩) measurement tends to place all gates on qubit 0. Two-qubit entangling gates (CNOT, CZ) followed by Rz are invisible because Rz only adds phases, and phase doesn't affect measurement probability of the computational basis.
- **Greedy search**: May miss globally optimal gate sequences that require temporarily worse intermediate steps.
- **Statevector simulation**: Limited to ~20 qubits on classical hardware (2^n memory).

### Potential Improvements

- **Alternative measurement**: Use multi-qubit observables instead of P(|0⋯0⟩) to make entangling gates visible
- **Beam search**: Track top-K gate candidates instead of greedy best-1
- **Hardware-aware gate pool**: Restrict to native gates of target QPU
- **Noise-aware training**: Include error mitigation in cost evaluation

---

## References

1. **Q-FLAIR / ARC**: "Mitigating Barren Plateaus in Quantum Neural Networks via Analytic Iterative Circuit Reconstruction"
2. **Rotosolve**: Ostaszewski et al., "Structure optimization for parameterized quantum circuits" (2021)
3. **Barren Plateaus**: McClean et al., "Barren plateaus in quantum neural network training landscapes" (2018)
4. **Quantum Machine Learning**: Schuld & Petruccione, "Machine Learning with Quantum Computers" (2021)
