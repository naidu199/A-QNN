# Adaptive Quantum Neural Network (A-QNN)

A novel approach to training Quantum Neural Networks that addresses the **barren plateau problem** through adaptive circuit construction and analytic iterative reconstruction.

## 🎯 Problem Statement

Traditional Quantum Neural Networks (QNNs) suffer from a major training issue known as the **barren plateau problem**, where:

- The optimization landscape becomes extremely flat as the number of qubits increases
- Gradients vanish exponentially, making gradient-based optimization ineffective
- Large-scale QNNs become essentially untrainable

## 💡 Our Solution

The Adaptive QNN implements a fundamentally different approach:

1. **Incremental Circuit Construction**: Instead of using a fixed circuit ansatz, gates are added one at a time based on their contribution to the learning objective.

2. **Analytic Parameter Estimation**: Optimal gate parameters are computed using closed-form solutions derived from the Fourier structure of quantum expectation values, eliminating the need for gradient descent.

3. **Measurement-Efficient Training**: Only O(1) measurements per parameter are required, independent of circuit size.

## 🔑 Key Innovations

| Feature           | Traditional VQC   | Adaptive QNN          |
| ----------------- | ----------------- | --------------------- |
| Circuit Structure | Fixed ansatz      | Adaptive construction |
| Parameter Updates | Gradient descent  | Analytic estimation   |
| Measurements      | O(parameters²)    | O(parameters)         |
| Barren Plateaus   | Severely affected | Mitigated by design   |

## 📁 Project Structure

```
A-QNN/
├── src/
│   ├── circuits/           # Quantum circuit construction
│   │   ├── quantum_gates.py    # Gate operations
│   │   ├── circuit_builder.py  # Adaptive circuit builder
│   │   └── encoding.py         # Data encoding strategies
│   │
│   ├── estimators/         # Parameter estimation methods
│   │   ├── analytic_estimator.py   # Core analytic estimation
│   │   ├── fourier_estimator.py    # Fourier-based methods
│   │   └── gradient_free.py        # Gradient-free optimizers
│   │
│   ├── models/             # QNN model implementations
│   │   ├── adaptive_qnn.py     # Main Adaptive QNN model
│   │   └── qnn_classifier.py   # Sklearn-compatible wrapper
│   │
│   ├── training/           # Training utilities
│   │   ├── trainer.py          # Training orchestration
│   │   └── callbacks.py        # Training callbacks
│   │
│   ├── data/               # Data handling
│   │   ├── preprocessing.py    # Data preprocessing
│   │   └── datasets.py         # Dataset loaders
│   │
│   ├── evaluation/         # Evaluation tools
│   │   ├── metrics.py          # Performance metrics
│   │   └── visualization.py    # Plotting utilities
│   │
│   └── utils/              # Utilities
│       ├── helpers.py          # Helper functions
│       └── config.py           # Configuration management
│
├── examples/               # Example scripts
│   ├── quick_start.py
│   ├── barren_plateau_analysis.py
│   └── dataset_comparison.py
│
├── train.py               # Main training script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models import AdaptiveQNN
from src.data import load_moons_quantum

# Load data
X_train, X_test, y_train, y_test = load_moons_quantum(
    n_samples=200, n_qubits=4
)

# Create and train model
model = AdaptiveQNN(
    n_qubits=4,
    n_classes=2,
    max_gates=30,
    encoding_type='angle'
)

model.fit(X_train, y_train, max_iterations=20)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Command Line Training

```bash
# Basic training
python train.py --dataset moons --n_qubits 4

# With comparison to baselines
python train.py --dataset iris --compare --save_plots

# Quick test
python train.py --preset quick_test --dataset circles
```

### Run Examples

```bash
# Quick start example
python examples/quick_start.py

# Barren plateau analysis
python examples/barren_plateau_analysis.py

# Dataset comparison
python examples/dataset_comparison.py
```

## 📊 Datasets

Built-in support for:

- **Moons**: Two interleaving half circles
- **Circles**: Two concentric circles
- **Iris**: Classic flower classification (binary)
- **XOR**: Quantum XOR pattern
- **Parity**: Parity classification

## 🔧 Configuration

Using preset configurations:

```python
from src.utils import get_preset

config = get_preset('medium')  # 'small', 'medium', 'large', 'quick_test'
```

Custom configuration:

```python
from src.utils import QNNConfig

config = QNNConfig(
    n_qubits=6,
    max_gates=50,
    encoding_type='iqp',
    shots=2048
)

config.save('my_config.yaml')
```

## 📈 Evaluation

```python
from src.evaluation import compute_metrics, barren_plateau_metrics

# Classification metrics
metrics = compute_metrics(y_true, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Barren plateau analysis
bp_metrics = barren_plateau_metrics(cost_history)
print(f"Barren Plateau Detected: {bp_metrics['barren_plateau_detected']}")
```

## 🛠️ Advanced Features

### Sklearn Integration

```python
from src.models import QNNClassifier
from sklearn.model_selection import cross_val_score

clf = QNNClassifier(n_qubits=4, max_gates=30)
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### Custom Encoding

```python
from src.circuits import DataEncoder

encoder = DataEncoder(
    n_qubits=4,
    encoding_type='iqp',  # 'angle', 'amplitude', 'iqp', 'dense_angle'
    reps=2
)
```

### Training Callbacks

```python
from src.training import QNNTrainer, TrainingConfig
from src.training.callbacks import EarlyStopping, ModelCheckpoint

trainer = QNNTrainer(model, TrainingConfig())
trainer.add_callback(EarlyStopping(patience=5))
trainer.add_callback(ModelCheckpoint('best_model.pkl'))
trainer.fit(X_train, y_train)
```

## 📚 References

This implementation is inspired by recent research on:

- Analytic gradient optimization in quantum circuits
- Rotosolve and Rotoselect algorithms
- Barren plateau mitigation strategies
- Adaptive variational quantum algorithms

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**Note**: This project requires a quantum computing environment. For best results, use Qiskit Aer simulator or compatible quantum hardware.
