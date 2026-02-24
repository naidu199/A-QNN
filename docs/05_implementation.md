# Chapter 5: Implementation

## 5.1  Data Preparation and Augmentation

### 5.1.1  Dataset Sources

The A-QNN system is evaluated on pre-processed benchmark datasets stored as comma-separated value (CSV) files inside the `Datasets/` directory.  Each dataset is already partitioned into a training split and a separate test split so that no data leakage can occur between the two phases.  The naming convention encodes every relevant parameter directly in the filename, for example `Xmnist_train_0_1_7x7_N_1000.csv` denotes the feature matrix for the MNIST 0-vs-1 task at 7 × 7 pixel resolution with 1 000 training samples.  The companion label file carries the `Y` prefix with the same suffix.

| Dataset | Pixel / Dim | Train N | Test N | Classes |
|---|---|---|---|---|
| MNIST 0 vs 1 (7×7) | 49 raw → 10 PCA | 1 000 | 1 000 | 2 |
| MNIST 0 vs 1 (14×14) | 196 raw → 10 PCA | 1 000 | 1 000 | 2 |
| MNIST 0 vs 1 (28×28) | 784 raw → 10 PCA | 1 000 | 1 000 | 2 |
| MNIST 3 vs 5 (7×7) | 49 raw → 10 PCA | 1 000 | 1 000 | 2 |
| MNIST PCA 10d | 10 | 11 552 | 1 902 | 2 |
| Two-Curves | 2 | 1 000 | 1 000 | 2 |
| Linearly Separable | 2 | 1 000 | 1 000 | 2 |
| Bars-and-Stripes | 4 | 1 000 | 200 | 2 |

### 5.1.2  The `DataPreprocessor` Class

All data transformations are managed by `DataPreprocessor` in [`src/data/preprocessing.py`](../src/data/preprocessing.py).  The class follows the scikit-learn `fit` / `transform` convention so that statistics computed on training data are applied identically to test data without leakage.

```python
preprocessor = DataPreprocessor(
    n_qubits=10,          # target qubit count caps PCA dimensions
    scaling='minmax',     # scales features to [0, π] for angle encoding
    reduce_dim=True,      # apply PCA when raw features > n_qubits
    target_range=(0, np.pi)
)
preprocessor.fit(X_train, y_train)
X_train_q = preprocessor.transform(X_train)
X_test_q  = preprocessor.transform(X_test)
```

**Scaling.**  The `'minmax'` mode wraps `sklearn.preprocessing.MinMaxScaler` and maps every feature linearly to the range $(0, \pi)$.  This is the natural domain for angle-encoded quantum circuits because the single-qubit rotation gate $R_Y(\theta)$ spans the full Bloch sphere meridian as $\theta$ traverses $(0, \pi)$.  The `'standard'` mode applies zero-mean unit-variance standardisation via `StandardScaler` and is available when features already span a consistent numerical range.

**Dimensionality reduction.**  When the raw feature count exceeds the number of available qubits — as is the case for every MNIST variant — `sklearn.decomposition.PCA` is applied to the scaled data and the output is compressed to exactly `n_qubits` principal components.  The PCA is fitted on the training split only.  The number of components is bounded by:

$$n_{\text{components}} = \min(n_{\text{qubits}},\ n_{\text{features}},\ n_{\text{training samples}})$$

This constraint ensures that the covariance matrix is always invertible and that the PCA decomposition is numerically stable.

**Label encoding.**  Class labels in the CSV files are integer-valued (0 or 1 for binary tasks, or arbitrary integers for multi-class problems).  `sklearn.preprocessing.LabelEncoder` normalises them to a contiguous zero-based integer range before they are passed to the cost function.

**Bias feature.**  The preprocessing pipeline appends a constant feature equal to 1.0 to every sample vector before it enters the circuit.  This gives the quantum circuit a static rotation degree of freedom that acts analogously to the bias term in a classical neuron, allowing the decision boundary to be shifted away from the origin.

### 5.1.3  Absence of Classical Augmentation

Classical image augmentation techniques (random crop, flip, colour jitter) are not applied because the input representation has already been collapsed to a small feature vector by PCA.  Spatial information is lost at that stage, so pixel-space augmentation would produce meaningless transformed samples.  The quantum nature of the simulator introduces no stochastic noise during training because the `AerSimulator` is used in `Statevector` mode (exact, no shot noise), so the reproducibility of every training run is guaranteed by the fixed `random_state` seed.

---

## 5.2  Model Training Process

### 5.2.1  ARC Training — Gate-by-Gate Construction

The ARC algorithm is implemented as the `ARCEstimator` class in [`src/estimators/arc_estimator.py`](../src/estimators/arc_estimator.py).  Training proceeds iteratively: at each step exactly one gate is appended to the growing circuit, and its optimal rotation angle is determined analytically without any gradient computation.

**Initialisation.**  The estimator is constructed with the following key hyper-parameters used in all primary experiments:

```python
arc = ARCEstimator(
    n_qubits=10,
    gate_list=['U1', 'U2', 'U3', 'H', 'X', 'Z'],
    subsample_size=100,           # training samples per gate iteration
    num_samples_classical=1001,   # θ grid points for cost minimisation
    max_gates=150,                # hard upper bound on circuit depth
    patience=1,                   # stop after first non-improving gate
    convergence_threshold=1e-8,
    n_jobs=-1                     # use all CPU cores via joblib
)
```

**Gate pool construction.**  `ARCGatePool._build_pool()` enumerates every combination of gate type and qubit position, producing a total of $|\text{gate\_list}| \times n_{\text{qubits}} = 6 \times 10 = 60$ candidate gate placements per iteration.  Each candidate is a descriptor vector of length $n_{\text{qubits}}$ where exactly one position carries the gate–feature string (e.g. `'U3_x'`) and all others carry the no-op placeholder `'111111'`.

**Sub-sampling.**  Evaluating all 60 candidates on the full training set of 1 000 samples per iteration would require $60 \times 1000 \times 3 = 180{,}000$ statevector simulations per gate step.  To keep wall-clock time tractable, a random sub-sample of `subsample_size = 100` training examples is drawn at the start of each iteration and held fixed for all candidate evaluations within that step.

**Rotosolve inner loop.**  For each candidate gate placed at angle $\theta$, the cost function $C(\theta)$ is a sinusoid of the form $C(\theta) = a \cos(\theta \cdot x_i - b) + c$.  The three parameters are recovered from only three circuit evaluations per training sample:

$$f_0 = P(\theta = 0),\quad f_+ = P(\theta = +\tfrac{\pi}{2}),\quad f_- = P(\theta = -\tfrac{\pi}{2})$$

$$c = \frac{f_+ + f_-}{2},\quad b = -\arctan_2\!\left(\frac{2(f_0 - c)}{f_+ - f_-}\right) + \frac{\pi}{2},\quad a = \sqrt{(f_0 - c)^2 + \frac{(f_+ - f_-)^2}{4}}$$

These computations are parallelised at the sample level using `joblib.Parallel(n_jobs=-1)` so that all CPU cores are utilised.

**Grid search for θ\*.**  Once the per-sample sinusoid parameters $\{(a_i, b_i, c_i)\}$ are collected, the aggregate log-loss across all sub-sampled training points is evaluated over a uniform grid of $1001$ candidate angles in $[-1, 1]$.  The angle $\theta^*$ that yields the minimum mean log-loss is selected as the optimal angle for that candidate gate.

**Gate selection and appending.**  After all 60 candidates have been evaluated, the gate–angle pair $(\text{gate}^*, \theta^*)$ with the globally lowest cost $C^*$ is selected.  If $C^* < C_{\text{prev}}$ (the cost before this step), the gate is appended to the circuit, the patience counter resets to zero, and training continues.  If no improvement is found, the patience counter increments.  Once the patience counter reaches the threshold (1 in primary experiments), training terminates early and the best circuit discovered so far is returned.

**Stopping criteria** (evaluated in order):
1. `patience` consecutive non-improving gate steps.
2. `max_gates` total gates appended.
3. `convergence_threshold` — improvement falls below $10^{-8}$.

### 5.2.2  Fixed Ansatz Baseline — COBYLA Optimisation

The `FixedAnsatzQNN` class in [`src/models/fixed_ansatz_qnn.py`](../src/models/fixed_ansatz_qnn.py) implements the conventional Variational Quantum Classifier (VQC) used as the performance baseline.

**Circuit structure.**  A Hadamard layer initialises all qubits into the $|+\rangle$ state.  Data features are encoded by angle-encoding $R_Y(x_i)$ rotations.  The trainable part repeats the following block for `n_layers` (default 3) times:
- $R_Y(\theta_j)$ rotation on every qubit.
- $R_Z(\phi_j)$ rotation on every qubit.
- Linear chain of CNOT gates for entanglement.

This produces $2 \times n_{\text{qubits}} \times n_{\text{layers}}$ variational parameters, all optimised simultaneously.

**Classical optimiser.**  All parameters are updated by `scipy.optimize.minimize` using the COBYLA (Constrained Optimisation By Linear Approximation) method with `max_iter = 200`.  COBYLA is a gradient-free simplex algorithm suited to noisy quantum function evaluations.  Unlike ARC, COBYLA must query the cost function for the full $2 \times n_{\text{qubits}} \times n_{\text{layers}}$ parameter vector in each iteration, making it vulnerable to the barren plateau phenomenon as circuit width grows.

---

## 5.3  Key Implementations

This section documents the most critical code components driving the A-QNN framework.

### 5.3.1  Sinusoidal Reconstruction — `determine_sine_curve()`

```python
def determine_sine_curve(f0, fp, fm):
    """
    Analytically recover a·cos(θ − b) + c from three samples.
    """
    c = 0.5 * (fp + fm)
    b = np.arctan2(2 * (f0 - c), fp - fm)
    a = np.sqrt((f0 - c) ** 2 + 0.25 * (fp - fm) ** 2)
    b = -b + np.pi / 2
    return a, b, c
```

This three-line function is the mathematical heart of ARC.  Given three cost measurements at fixed angles, it recovers the full sinusoidal landscape in $O(1)$ time — eliminating gradient computation entirely.

### 5.3.2  Gate Circuit Builder — `build_gate_circuit()`

The function [`build_gate_circuit()`](../src/estimators/arc_estimator.py) translates a string gate descriptor into a Qiskit `QuantumCircuit` object.  The mapping between descriptor tokens and Qiskit gate calls is:

| Token | Qiskit Gate | Note |
|---|---|---|
| `U1` | `circuit.rz(θ·x, q[j])` | Data-dependent Rz |
| `U2` | `circuit.rx(θ·x, q[j])` | Data-dependent Rx |
| `U3` | `circuit.ry(θ·x, q[j])` | Data-dependent Ry |
| `H`  | `circuit.h(q[j])` | No rotation; static |
| `X`  | `circuit.cx(q[j-1], q[j])`  + `circuit.rz(θ·x, q[j])` | CNOT + Rz |
| `Z`  | `circuit.cz(q[j-1], q[j])`  + `circuit.rz(θ·x, q[j])` | CZ + Rz |
| `Xn` | `circuit.cx(q[j-2], q[j])` + `circuit.rz(θ·x, q[j])` | Skip-CNOT |
| `Zn` | `circuit.cz(q[j-2], q[j])` + `circuit.rz(θ·x, q[j])` | Skip-CZ |

Python's negative-index wrap-around (`q[-1]` equals the last qubit) is intentionally used so that the two-qubit gate at qubit 0 wraps around to qubit n−1, forming a circular entanglement topology without conditional checks.

### 5.3.3  Parallel Sample Worker — `_compute_abc_for_sample()`

```python
def _compute_abc_for_sample(base_circ_bound, test_gates_3, n_qubits):
    """Module-level worker: compute (a,b,c) for one training sample."""
    qubits = list(range(n_qubits))
    expect = [0.0, 0.0, 0.0]
    for kt in range(3):
        tc = base_circ_bound.copy()
        tc.compose(test_gates_3[kt], qubits=qubits, inplace=True)
        rem = {p: 0.0 for p in tc.parameters}
        if rem:
            tc = tc.assign_parameters(rem)
        expect[kt] = Statevector(tc).probabilities()[0]
    return determine_sine_curve(expect[0], expect[1], expect[2])
```

This function is defined at module level (not as a class method) so that Python's `multiprocessing` serialiser (`pickle`) can locate it across worker processes.  Each worker receives a pre-bound base circuit and three pre-built test-angle circuits, computes the three statevector probabilities, and returns the $(a, b, c)$ triple for that sample.  `joblib.Parallel(n_jobs=-1, prefer='threads')` dispatches all samples concurrently.

### 5.3.4  Pre-binding Optimisation

Before the inner candidate loop begins, the current accumulated circuit is bound to each sub-sampled training feature vector using `circuit.assign_parameters(param_dict)`.  These pre-bound base circuits are computed once and reused for every gate candidate, avoiding redundant parameter substitutions.  The three test-angle circuits (at $\theta = 0, +\pi/2, -\pi/2$) are also built once per candidate and shared across samples, further reducing object creation overhead.

### 5.3.5  Comparison Driver — `compare_qnn.py`

The top-level script [`compare_qnn.py`](../compare_qnn.py) orchestrates the full comparison workflow:

1. **Dataset loading** — resolves the file path via `_find_data_dir()` and reads NumPy arrays from CSV.
2. **Preprocessing** — calls `DataPreprocessor.fit_transform()` with PCA capping to `n_qubits`.
3. **ARC training** — instantiates `ARCEstimator`, calls `fit()`, records cost history and wall time.
4. **Baseline training** — instantiates `FixedAnsatzQNN`, calls `fit()` with COBYLA.
5. **Evaluation** — computes accuracy, F1-score, AUC-ROC, and confusion matrix via `src/evaluation/metrics.py`.
6. **Result export** — serialises all metrics to `results/comparison_results.json` and a human-readable `results/comparison_report.txt`.

The script exposes `--ibm_token` for optional execution on real IBM Quantum hardware and `--barren_plateau` for variance-of-cost analysis across random initialisations.

### 5.3.6  Multi-class Extension

`AdaptiveQNN` in [`src/models/adaptive_qnn.py`](../src/models/adaptive_qnn.py) supports multi-class classification through a One-vs-Rest (OvR) decomposition.  When `n_classes > 2`, a separate `ARCEstimator` is trained for each class against all others, and the output probabilities from each binary classifier are assembled into a probability vector.  The final predicted class is the index of the highest probability:

```python
model = AdaptiveQNN(n_qubits=10, n_classes=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 5.4  Quantum Backend Setup

### 5.4.1  Simulation Engine — Qiskit Aer

All training and evaluation in the primary experiments are performed using `AerSimulator` from the `qiskit-aer` package (v ≥ 0.13.0).  The simulator is instantiated with default settings inside `AdaptiveQNN`:

```python
from qiskit_aer import AerSimulator
self.backend = backend or AerSimulator()
```

For cost-function evaluations during the Rotosolve inner loop, the `Statevector` class from `qiskit.quantum_info` is used directly rather than the shot-based `AerSimulator.run()` path:

```python
probability_zero = Statevector(circuit).probabilities()[0]
```

This returns the exact ground-state probability $|\langle 0|U(x)|0\rangle|^2$ without any shot noise, which is essential for the sinusoidal reconstruction step to be numerically stable.  Stochastic shot noise would corrupt the three-point measurements $(f_0, f_+, f_-)$ and produce an inaccurate $(a, b, c)$ estimate.

### 5.4.2  Statevector vs Shot-based Modes

| Mode | API | Noise | Speed | Used For |
|---|---|---|---|---|
| Statevector | `Statevector(circ).probabilities()` | None (exact) | Fast (O(2ⁿ) memory) | ARC Rotosolve training |
| Shot-based | `AerSimulator.run(circ, shots=1024)` | Binomial shot noise | Slower | Post-training evaluation & IBM backend emulation |

During the post-training accuracy measurement reported in the comparison results, shot-based simulation with `shots = 1024` is re-enabled to emulate the behaviour of real quantum hardware.  This ensures that the reported test accuracy reflects realistic hardware noise levels rather than ideal simulation.

### 5.4.3  Circuit Transpilation

Before execution on any backend (simulated or real), Qiskit's `transpile()` function maps the logical circuit to the target device's native gate set and qubit connectivity:

```python
from qiskit import transpile
transpiled = transpile(circuit, backend=self.backend,
                       optimization_level=1)
```

`optimization_level=1` applies lightweight gate cancellation and basis translation.  Higher optimisation levels are not used during training because the circuit changes at every ARC iteration and the transpilation overhead would dominate the runtime.

### 5.4.4  IBM Quantum Integration (Optional)

The `compare_qnn.py` script exposes an `--ibm_token` command-line argument for users who wish to run experiments on real IBM Quantum hardware via the IBM Quantum Runtime service.  When provided, the token is loaded from an `.env` file via `python-dotenv` and used to initialise a `QiskitRuntimeService` session.  The `SamplerV2` primitive then replaces the local `AerSimulator` for all circuit executions.  This mode is optional and does not affect the simulation-based results reported in this project.

### 5.4.5  Dependency Summary

The complete environment is defined in [`requirements.txt`](../requirements.txt):

| Package | Minimum Version | Role |
|---|---|---|
| `qiskit` | ≥ 0.45.0 | Circuit construction, parameter binding |
| `qiskit-aer` | ≥ 0.13.0 | Statevector and shot-based simulation |
| `qiskit-machine-learning` | ≥ 0.7.0 | QNN primitives and VQC utilities |
| `numpy` | ≥ 1.24.0 | Numerical arrays and linear algebra |
| `scipy` | ≥ 1.11.0 | COBYLA optimiser, grid search utilities |
| `scikit-learn` | ≥ 1.3.0 | PCA, scalers, label encoder, metrics |
| `joblib` | bundled with scikit-learn | Parallel Rotosolve workers |
| `matplotlib` / `seaborn` | ≥ 3.7.0 / 0.12.0 | Result visualisation |

All packages are installable via `pip install -r requirements.txt`.  No custom C extensions or special hardware drivers are required for the simulation path, making the environment fully reproducible on any standard Python ≥ 3.10 installation.
