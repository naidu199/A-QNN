"""Debug training to understand why no improvement is found."""
from src.models import AdaptiveQNN
from src.data import load_breast_cancer_quantum
import numpy as np

np.random.seed(42)
X_train, X_test, y_train, y_test = load_breast_cancer_quantum(n_qubits=4)

# Use small subset for debugging
X_small = X_train[:20]
y_small = y_train[:20]

print("Creating model...")
model = AdaptiveQNN(n_qubits=4, n_classes=2, encoding_type='angle', max_gates=30)
model.build_initial_circuit(n_features=4)

# Create cost function
X_processed = model.data_encoder.preprocess_data(X_small, method='minmax')
cost_fn = model._create_cost_function(X_processed, y_small)

# Check initial cost
initial_cost = cost_fn(model.circuit, model.trained_params)
print(f"Initial cost: {initial_cost}")

# Test one gate manually
print("\nTesting first gate from pool...")
trial_builder = model.circuit_builder.copy()
gate = model.gate_pool[0]
print(f"Gate: {gate}")

param, _ = trial_builder.add_adaptive_gate(gate)
trial_circuit = trial_builder.get_circuit()

print(f"New param: {param}")
print(f"Trial circuit:\n{trial_circuit}")

# Try landscape analysis
print("\nAnalyzing landscape...")
result = model.fourier_estimator.analyze_landscape(
    trial_circuit, param, cost_fn, {}, n_samples=5
)
print(f"Optimal theta: {result['optimal_theta']:.4f}")
print(f"Optimal cost: {result['optimal_cost']:.4f}")
print(f"Cost improvement: {initial_cost - result['optimal_cost']:.4f}")

# Check a few more gates
print("\n\nTesting multiple gates...")
for i, gate in enumerate(model.gate_pool[:10]):
    trial_builder = model.circuit_builder.copy()
    param, _ = trial_builder.add_adaptive_gate(gate)
    trial_circuit = trial_builder.get_circuit()

    if param is not None:
        result = model.fourier_estimator.analyze_landscape(
            trial_circuit, param, cost_fn, {}, n_samples=5
        )
        improvement = initial_cost - result['optimal_cost']
        print(f"Gate {i}: {gate['type']} on {gate['qubits']} -> improvement: {improvement:.4f}")
