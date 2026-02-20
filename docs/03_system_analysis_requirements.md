# Chapter 3: System Analysis and Requirements

This chapter establishes the formal analytical and practical foundations of the Adaptive Quantum Neural Network (A-QNN) project. It begins by precisely formulating the supervised classification problem in the quantum computing context — covering both binary and multi-class settings — describes each dataset used in evaluation along with its preprocessing pipeline, assesses the feasibility of the project across technical and economic dimensions, and concludes with a complete specification of the hardware and software environment.

---

## 3.1 Problem Formulation

### 3.1.1 Supervised Classification as a Quantum Learning Task

The core task addressed by this project is **supervised classification**: given a labeled training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ where $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector and $y_i \in \{0, 1, \ldots, C-1\}$ is a class label from $C$ possible classes, the objective is to learn a function $f: \mathbb{R}^d \to \{0, 1, \ldots, C-1\}$ that correctly predicts the label of unseen test samples.

The A-QNN framework supports both **binary** ($C = 2$) and **multi-class** ($C > 2$) settings. For the binary case, the framework directly trains a single ARC circuit whose measurement probability serves as the decision score. For multi-class tasks, the `AdaptiveQNN` model employs a **One-vs-Rest (OvR)** decomposition: one ARC circuit is trained per class, and the class whose circuit reports the highest measurement probability is selected as the final prediction. This stratified design means the core ARC algorithm is class-count agnostic — it always trains a single binary discriminator — and multi-class capability scales linearly in the number of classes.

In the quantum computing context, each binary discriminator is realized by a parameterized quantum circuit $U(\boldsymbol{\theta}, \mathbf{x})$ acting on $n$ qubits. The circuit encodes the input $\mathbf{x}$ through rotation angles and produces a measurement outcome — specifically, the probability $P(|0\cdots0\rangle)$ of observing the all-zero computational basis state — which serves as the decision score:

$$\hat{y} = \begin{cases} 0 & \text{if } P(|0\cdots0\rangle) > 0.5 \\ 1 & \text{if } P(|0\cdots0\rangle) \leq 0.5 \end{cases}$$

The cost function minimized during training is the binary cross-entropy (log-loss):

$$C(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

where $\hat{p}_i = P(|0\cdots0\rangle \mid \mathbf{x}_i, \boldsymbol{\theta})$ is the predicted probability for sample $i$. For multi-class inference under OvR, the final label is $\hat{y} = \arg\max_{c} \hat{p}_i^{(c)}$ where $\hat{p}_i^{(c)}$ is the score from the $c$-th class circuit.

### 3.1.2 The ARC Optimization Problem

Unlike traditional VQC training where the circuit structure is fixed and all parameters $\boldsymbol{\theta}$ are optimized simultaneously, the ARC algorithm reformulates training as a **sequential structure-and-parameter search**:

At each iteration $t$, given a current circuit $U^{(t)}$, find the gate $g^* \in \mathcal{G}$ (selected from gate pool $\mathcal{G}$), the feature index $f^* \in \{0, \ldots, d\}$, and the rotation parameter $\theta^*$ that jointly minimize the cost:

$$(g^*, f^*, \theta^*) = \arg\min_{g \in \mathcal{G},\, f,\, \theta} \; C\!\left(U^{(t)} \oplus g(\theta \cdot x_f)\right)$$

The parameter $\theta^*$ is determined analytically using the Rotosolve principle (Section 3.1.3), reducing a potentially expensive global search to a closed-form computation. The selected gate is then appended permanently to the circuit and the process repeats until a patience-based convergence criterion is met ($k$ consecutive iterations without cost improvement, where $k$ is the patience hyperparameter).

Additionally, a **bias feature** $x_{d} = \pi$ (constant for all samples) is appended to the feature matrix. This ensures that the circuit can always add a gate that operates as a qubit bias regardless of the input data distribution.

### 3.1.3 Analytic Parameter Estimation (Rotosolve)

For any single rotation gate $R(\theta \cdot x_f)$ appended to a fixed base circuit, the expectation value of $P(|0\cdots0\rangle)$ is a sinusoidal function of $\theta$:

$$P(\theta) = a_i \cos(\theta \cdot x_{i,f} - b_i) + c_i \quad \text{for each sample } i$$

where $a_i$, $b_i$, $c_i$ are constants depending only on the base circuit and sample $\mathbf{x}_i$ (not on $\theta$). These constants are recovered from exactly three circuit evaluations at test angles $\theta_0 = 0$, $\theta_+ = \pi/2$, $\theta_- = -\pi/2$:

$$c_i = \frac{f_{i,+} + f_{i,-}}{2}, \quad b_i = -\arctan2\!\left(2(f_{i,0} - c_i),\, f_{i,+} - f_{i,-}\right) + \frac{\pi}{2}$$

$$a_i = \sqrt{(f_{i,0} - c_i)^2 + \frac{(f_{i,+} - f_{i,-})^2}{4}}$$

The reconstructed per-sample cost landscape $C(\theta)$ is then evaluated over a fine grid of 1000 candidate angles in $[-1, 1]$ to identify $\theta^*$ analytically — with no gradient computation required.

### 3.1.4 Number of Qubits and Feature Mapping

The number of qubits $n$ equals the number of features used by the circuit. When the raw feature dimensionality $d$ exceeds a specified qubit budget $n_{\text{qubits}}$, Principal Component Analysis (PCA) is applied to project $\mathbf{x} \in \mathbb{R}^d$ down to $\mathbb{R}^{n_{\text{qubits}}}$. Each feature occupies one qubit in the encoding layer, meaning the qubit count determines the maximum expressible data resolution.

---

## 3.2 Datasets Description and Preprocessing

Six benchmark datasets are used in this project, covering a range of data geometries, dimensionalities, class counts, and real-world complexity levels. All datasets are stored as pre-split CSV files in the `Datasets/` directory, with separate `X` (feature) and `Y` (label) files for training and test sets. While the current evaluation focuses on binary tasks to enable a controlled comparison with the fixed-ansatz baseline, the framework natively supports multi-class datasets through One-vs-Rest decomposition.

### 3.2.1 Dataset Overview

| Dataset            | Train Samples | Test Samples | Raw Features | Qubits Used | Task                    |
| ------------------ | ------------- | ------------ | ------------ | ----------- | ----------------------- |
| Linearly Separable | 1,000         | 1,000        | 4            | 4           | Synthetic linear        |
| Two-Curves         | 1,000         | 1,000        | 8            | 4–8         | Synthetic nonlinear     |
| Bars and Stripes   | 1,000         | 200          | 16           | 4–16        | Binary pixel pattern    |
| MNIST 0 vs 1 (7×7) | 1,000         | 1,000        | 49           | 4–10        | Handwritten digit       |
| MNIST 3 vs 5 (7×7) | 1,000         | 1,000        | 49           | 4–10        | Handwritten digit       |
| MNIST PCA (10d)    | 11,552        | 1,902        | 11           | 11          | Handwritten digit (PCA) |

### 3.2.2 Dataset Descriptions

**Linearly Separable** is a synthetically generated dataset where the decision boundary is a hyperplane. With only 4 features, it serves as a baseline sanity check to verify that the ARC circuit can at minimum recover a linear classifier. It is used in binary mode here, though the dataset generator supports arbitrary class counts.

**Two-Curves** is a synthetic dataset consisting of two interleaved curves in 8-dimensional space, requiring a nonlinear decision boundary. It tests whether the adaptive circuit can construct sufficient expressibility to handle non-trivial geometries without over-parameterization.

**Bars and Stripes** is a 4×4 binary pixel pattern dataset with 16 raw features encoded as flattened pixel intensities. Samples are labeled based on whether the pattern consists of uniform horizontal bars (class 0) or vertical stripes (class 1). This dataset tests the ability of QNNs to learn spatial structure from pixel-level features.

**MNIST 0 vs 1 (7×7)** consists of downsampled 7×7 grayscale images of handwritten digits 0 and 1, yielding 49 raw features per sample. This is the primary evaluation dataset of the project, run with 10 qubits and PCA reduction from 49 to 10 features. It represents a real-world image classification task at a scale tractable for quantum simulation.

**MNIST 3 vs 5 (7×7)** follows the same format as 0 vs 1 but classifies the more visually similar digit pair 3 and 5, posing a harder classification problem to probe the discriminative capacity of ARC circuits.

**MNIST PCA (10d)** is a larger pre-PCA-reduced version of the full MNIST dataset (all digits, binary filtered to a specific pair), with 11 features (10 PCA components plus bias). Its larger training set (11,552 samples) tests scalability of the ARC training loop.

### 3.2.3 Preprocessing Pipeline

All datasets pass through the following standardized preprocessing steps implemented in `compare_qnn.py` and `src/data/preprocessing.py`:

**Step 1 — Load from CSV.** Feature matrices $X$ and label vectors $y$ are read from comma-separated value files using `numpy.loadtxt`.

**Step 2 — Label normalization.** Labels are mapped to the binary set $\{0, 1\}$. Datasets using $\{-1, +1\}$ encoding (common in SVM literature) are converted as $y \leftarrow \mathbb{1}[y = +1]$.

**Step 3 — Dimensionality reduction (conditional).** If the number of raw features $d$ exceeds the specified qubit count $n_{\text{qubits}}$, PCA is fitted on the training set and applied to both train and test sets to produce $n_{\text{qubits}}$ components. The explained variance ratio is logged at runtime. For the primary experiment (MNIST 0 vs 1, 7×7, 10 qubits), PCA reduces 49 features to 10, retaining approximately 80–85% of total variance.

**Step 4 — Feature scaling.** All features are scaled to the range $[0, \pi]$ using `MinMaxScaler` from scikit-learn, fitted exclusively on the training set to prevent data leakage. This range is chosen to match the natural angular domain of quantum rotation gates ($R_x$, $R_y$, $R_z$), ensuring full utilization of the Bloch sphere.

**Step 5 — Bias feature appension.** A constant feature column $x_d = \pi$ is appended to the scaled feature matrix prior to ARC training. This bias feature provides the adaptive circuit with a constant-rotation baseline gate analogous to the bias term in a classical linear model.

---

## 3.3 Feasibility Analysis

### 3.3.1 Technical Feasibility

The technical feasibility of the A-QNN project was assessed across three dimensions: computational, algorithmic, and software.

**Computational Feasibility.** The ARC algorithm relies on statevector simulation via Qiskit's `Statevector` class, which stores $2^n$ complex amplitudes and applies unitary transformations exactly. For $n = 10$ qubits, this requires $2^{10} = 1024$ complex numbers — approximately 16 KB of memory — well within the capabilities of any modern workstation. Simulations up to $n = 20$ qubits (approximately 16 MB of complex amplitudes) are readily feasible on consumer-grade hardware with 8 GB RAM. The primary computational cost is the number of statevector evaluations: for a gate pool of $G$ candidates, $F$ features, and training subsample size $S$, each ARC iteration requires $3 \times G \times S$ statevector computations (the factor of 3 comes from the three Rotosolve test angles). For the primary experiment ($G = 60$, $S = 100$, $n = 10$), this yields 18,000 statevector evaluations per gate — a tractable workload taking approximately 20–30 seconds per gate on a modern CPU with Joblib parallelism. For multi-class tasks with $C$ classes, the OvR strategy trains $C$ independent ARC circuits, scaling training time linearly with class count while keeping per-circuit complexity unchanged.

**Algorithmic Feasibility.** The Rotosolve analytic parameter estimation is mathematically exact for single-qubit rotation gates, and its correctness has been validated in the published Q-FLAIR paper (arXiv:2510.03389). The greedy gate selection strategy is a well-studied approach in adaptive variational algorithms (ADAPT-VQE) with documented convergence on practical instances. The patience-based stopping criterion is a standard early-stopping technique from classical machine learning that has been shown empirically to prevent overfitting in greedy search procedures.

**Software Feasibility.** All required software components are open-source and actively maintained. PennyLane and Qiskit provide mature quantum circuit simulation backends with well-documented Python APIs. Scikit-learn, NumPy, and SciPy provide all necessary classical machine learning and numerical computing utilities. Joblib provides CPU-level parallelism with minimal overhead. The entire software stack runs on standard Python 3.9+ environments with no proprietary dependencies, making the project fully reproducible on any machine with the required packages installed.

**Conclusion:** The project is fully technically feasible. All algorithms, simulations, and evaluations have been successfully executed, as evidenced by the experimental results reported in Chapter 5.

### 3.3.2 Economic Feasibility

The A-QNN project is economically feasible from both development and deployment perspectives.

**Development Cost.** The entire project was developed using open-source software libraries at zero licensing cost. The development environment (Python, VS Code, Qiskit, PennyLane, NumPy, scikit-learn) is freely available. No quantum hardware access was required for the primary simulation-based experiments; all training and evaluation runs on standard CPU hardware available to any university or research institution.

**Computational Cost.** Training the ARC model on the primary MNIST 0 vs 1 (10-qubit) benchmark requires approximately 5–8 minutes of CPU time on a modern multi-core processor with Joblib parallelism enabled. This is well within the budget of any personal or institutional computing resource. The comparative fixed-ansatz COBYLA baseline requires approximately 1–2 minutes per run, also at negligible cost.

**Quantum Hardware Cost.** While the project is designed for classical simulation, the resulting circuits are compatible with real quantum hardware. IBM Quantum Network provides free access to small ($< 5$-qubit) quantum processors through its public plan, and access to larger devices through institutional plans. Deployment on real hardware would incur only the time cost of circuit submission and queue wait times, not financial cost for academic users.

**Maintenance Cost.** The modular codebase, structured around well-separated `src/` modules with clear interfaces, minimizes long-term maintenance burden. Adding new datasets, gate types, or evaluation metrics requires modification of only localized components without restructuring the core training loop.

**Conclusion:** The project requires no financial investment beyond standard computing infrastructure, making it economically feasible for any academic or research setting.

---

## 3.4 Hardware and Software Specifications

### 3.4.1 Hardware Specifications

The following hardware configuration was used for all development, training, and evaluation experiments in this project:

| Component        | Specification                                                          |
| ---------------- | ---------------------------------------------------------------------- |
| Processor        | Multi-core x86-64 CPU (Intel Core i5/i7 or equivalent)                 |
| RAM              | Minimum 8 GB (16 GB recommended for $n > 15$ qubits)                   |
| Storage          | 10 GB free disk space (for datasets, results, and virtual environment) |
| Operating System | Windows 10/11 (64-bit)                                                 |
| GPU              | Not required (all computation is CPU-based statevector simulation)     |

For qubit counts up to 10, statevector simulation runs comfortably within 1 GB of RAM. For experimentation with up to 20 qubits, 8–16 GB RAM is recommended due to the exponential memory scaling of the state vector ($2^{20}$ complex128 values ≈ 16 MB).

### 3.4.2 Software Specifications

The complete software stack is specified in `requirements.txt` and summarized below:

**Core Quantum Computing Libraries**

| Package                   | Version  | Purpose                                                               |
| ------------------------- | -------- | --------------------------------------------------------------------- |
| `qiskit`                  | ≥ 0.45.0 | Quantum circuit construction, gate operations, statevector simulation |
| `qiskit-aer`              | ≥ 0.13.0 | High-performance circuit simulation backend                           |
| `qiskit-machine-learning` | ≥ 0.7.0  | Quantum kernel methods and VQC utilities                              |
| `pennylane`               | ≥ 0.35.0 | Quantum differentiable programming (used for gradient reference)      |

**Scientific Computing and Machine Learning**

| Package        | Version  | Purpose                                                            |
| -------------- | -------- | ------------------------------------------------------------------ |
| `numpy`        | ≥ 1.24.0 | Array operations, vectorized log-loss, PCA feature reduction       |
| `scipy`        | ≥ 1.11.0 | Numerical optimization routines                                    |
| `scikit-learn` | ≥ 1.3.0  | PCA, MinMaxScaler, metrics (accuracy, F1, AUC-ROC), COBYLA wrapper |
| `pandas`       | ≥ 2.0.0  | Results tabulation and export                                      |

**Parallelism and Performance**

| Package  | Version | Purpose                                                           |
| -------- | ------- | ----------------------------------------------------------------- |
| `joblib` | ≥ 1.3.0 | Multi-core parallel execution of per-sample Rotosolve evaluations |

**Visualization and Reporting**

| Package      | Version  | Purpose                                                         |
| ------------ | -------- | --------------------------------------------------------------- |
| `matplotlib` | ≥ 3.7.0  | Cost history plots, accuracy bar charts, circuit visualizations |
| `seaborn`    | ≥ 0.12.0 | Statistical comparison plots                                    |

**Development and Testing**

| Package         | Version | Purpose                                              |
| --------------- | ------- | ---------------------------------------------------- |
| `pytest`        | ≥ 7.4.0 | Unit and integration testing                         |
| `pyyaml`        | ≥ 6.0.0 | Configuration file management                        |
| `jupyter`       | ≥ 1.0.0 | Interactive notebook-based exploration               |
| `python-dotenv` | ≥ 1.0.0 | Environment variable management (IBM token handling) |

**Python Version:** Python 3.9 or higher is required. Python 3.11 was used for all primary experiments.

**Development Environment:** Visual Studio Code with the Python and Pylance extensions provided the primary development environment. All scripts are executable from the command line and require no IDE-specific tooling.

---

_References for this chapter are consolidated in the project bibliography._
