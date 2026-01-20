"""
Adaptive Circuit Builder
=========================

This module implements the adaptive quantum circuit construction strategy.
Unlike traditional fixed ansatz approaches, the circuit is built incrementally
by adding gates that maximize the training objective.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from typing import List, Tuple, Optional, Dict, Any, Callable
from copy import deepcopy


from .quantum_gates import QuantumGateSet, create_adaptive_gate_pool


class AdaptiveCircuitBuilder:
    """
    Builds quantum circuits adaptively by incrementally adding gates.

    This builder implements the core idea of adaptive QNN: instead of using
    a fixed circuit structure, gates are added one at a time based on their
    contribution to the learning objective. This approach:

    1. Reduces circuit depth by only including necessary gates
    2. Mitigates barren plateaus by ensuring each gate is meaningful
    3. Enables deterministic parameter estimation through analytic methods

    Attributes:
        n_qubits: Number of qubits in the circuit
        n_layers: Current number of layers in the circuit
        circuit: The quantum circuit being built
        parameters: List of all parameters in the circuit
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int = 2,
        encoding_type: str = 'amplitude',
        measurement_type: str = 'expectation'
    ):
        """
        Initialize the adaptive circuit builder.

        Args:
            n_qubits: Number of qubits
            n_classes: Number of output classes for classification
            encoding_type: Data encoding strategy ('amplitude', 'angle', 'iqp')
            measurement_type: How to extract output ('expectation', 'probability')
        """
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.encoding_type = encoding_type
        self.measurement_type = measurement_type

        self.circuit: Optional[QuantumCircuit] = None
        self.parameters: List[Parameter] = []
        self.gate_set = QuantumGateSet(n_qubits)
        self.gate_pool = create_adaptive_gate_pool(n_qubits)

        self.layer_info: List[Dict[str, Any]] = []
        self.n_layers = 0

        # Initialize empty circuit
        self._initialize_circuit()

    def _initialize_circuit(self) -> None:
        """Initialize the base quantum circuit."""
        self.circuit = QuantumCircuit(self.n_qubits, name='AdaptiveQNN')
        self.gate_set.reset()
        self.parameters.clear()
        self.layer_info.clear()
        self.n_layers = 0

    def add_encoding_layer(
        self,
        data_params: Optional[ParameterVector] = None
    ) -> ParameterVector:
        """
        Add data encoding layer to the circuit.

        This layer encodes classical data into quantum states. Different
        encoding strategies provide different expressibility:

        - 'amplitude': Encodes data in state amplitudes (requires 2^n features)
        - 'angle': Encodes each feature as a rotation angle
        - 'iqp': IQP-style encoding with product feature maps

        Args:
            data_params: Optional parameter vector for data encoding

        Returns:
            ParameterVector used for data encoding
        """
        if self.encoding_type == 'angle':
            return self._add_angle_encoding(data_params)
        elif self.encoding_type == 'amplitude':
            return self._add_amplitude_encoding(data_params)
        elif self.encoding_type == 'iqp':
            return self._add_iqp_encoding(data_params)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def _add_angle_encoding(
        self,
        data_params: Optional[ParameterVector] = None
    ) -> ParameterVector:
        """Add angle encoding layer."""
        n_features = self.n_qubits

        if data_params is None:
            data_params = ParameterVector('x', n_features)

        # Apply Hadamard to create superposition
        for i in range(self.n_qubits):
            self.circuit.h(i)

        # Encode each feature as RY rotation
        for i, param in enumerate(data_params[:self.n_qubits]):
            self.circuit.ry(param, i)

        self.layer_info.append({
            'type': 'encoding',
            'encoding_type': 'angle',
            'n_features': n_features
        })

        return data_params

    def _add_amplitude_encoding(
        self,
        data_params: Optional[ParameterVector] = None
    ) -> ParameterVector:
        """Add amplitude encoding layer (placeholder for state preparation)."""
        n_features = 2 ** self.n_qubits

        if data_params is None:
            data_params = ParameterVector('x', n_features)

        # Note: Full amplitude encoding requires complex state preparation
        # For simplicity, we use a parametric approximation
        for i in range(self.n_qubits):
            self.circuit.h(i)
            if i < len(data_params):
                self.circuit.ry(data_params[i], i)

        self.layer_info.append({
            'type': 'encoding',
            'encoding_type': 'amplitude',
            'n_features': n_features
        })

        return data_params

    def _add_iqp_encoding(
        self,
        data_params: Optional[ParameterVector] = None
    ) -> ParameterVector:
        """Add IQP-style encoding with product feature maps."""
        n_features = self.n_qubits

        if data_params is None:
            data_params = ParameterVector('x', n_features)

        # First layer: Hadamard
        for i in range(self.n_qubits):
            self.circuit.h(i)

        # Single-qubit rotations
        for i, param in enumerate(data_params[:self.n_qubits]):
            self.circuit.rz(param, i)

        # Two-qubit interactions for feature products
        for i in range(self.n_qubits - 1):
            if i < len(data_params) and i + 1 < len(data_params):
                self.circuit.cx(i, i + 1)
                # Product feature: x_i * x_{i+1}
                self.circuit.rz(data_params[i] * data_params[i + 1], i + 1)
                self.circuit.cx(i, i + 1)

        self.layer_info.append({
            'type': 'encoding',
            'encoding_type': 'iqp',
            'n_features': n_features
        })

        return data_params

    def add_variational_layer(
        self,
        layer_type: str = 'standard',
        entanglement: str = 'linear'
    ) -> List[Parameter]:
        """
        Add a variational layer to the circuit.

        Args:
            layer_type: Type of variational layer
                - 'standard': RY-RZ rotations + entanglement
                - 'efficient': Single rotation + entanglement
                - 'strongly_entangling': Multiple rotation types + full entanglement
            entanglement: Entanglement pattern

        Returns:
            List of parameters added in this layer
        """
        layer_params = []

        if layer_type == 'standard':
            params = self.gate_set.add_variational_block(
                self.circuit,
                rotation_gates=['ry', 'rz'],
                entanglement=entanglement
            )
            layer_params.extend(params)

        elif layer_type == 'efficient':
            params = self.gate_set.add_rotation_layer(self.circuit, 'ry')
            layer_params.extend(params)
            self.gate_set.add_entangling_layer(self.circuit, pattern=entanglement)

        elif layer_type == 'strongly_entangling':
            for gate_type in ['rx', 'ry', 'rz']:
                params = self.gate_set.add_rotation_layer(self.circuit, gate_type)
                layer_params.extend(params)
            self.gate_set.add_entangling_layer(self.circuit, pattern='full')

        self.parameters.extend(layer_params)
        self.n_layers += 1

        self.layer_info.append({
            'type': 'variational',
            'layer_type': layer_type,
            'entanglement': entanglement,
            'n_params': len(layer_params)
        })

        return layer_params

    def add_single_gate(
        self,
        gate_type: str,
        qubits: List[int],
        parameter: Optional[Parameter] = None
    ) -> Optional[Parameter]:
        """
        Add a single gate to the circuit (for adaptive construction).

        This method is used during the adaptive construction phase to
        add individual gates based on their contribution to the objective.

        Args:
            gate_type: Type of gate to add
            qubits: Target qubit(s)
            parameter: Optional parameter for parameterized gates

        Returns:
            Parameter if the gate is parameterized, None otherwise
        """
        param = None

        if gate_type in ['rx', 'ry', 'rz']:
            if parameter is None:
                parameter = self.gate_set.create_parameter()
            getattr(self.circuit, gate_type)(parameter, qubits[0])
            param = parameter
            self.parameters.append(parameter)

        elif gate_type == 'cx':
            self.circuit.cx(qubits[0], qubits[1])

        elif gate_type == 'cz':
            self.circuit.cz(qubits[0], qubits[1])

        elif gate_type in ['crx', 'cry', 'crz']:
            if parameter is None:
                parameter = self.gate_set.create_parameter()
            getattr(self.circuit, gate_type)(parameter, qubits[0], qubits[1])
            param = parameter
            self.parameters.append(parameter)

        self.layer_info.append({
            'type': 'single_gate',
            'gate_type': gate_type,
            'qubits': qubits,
            'parameterized': param is not None
        })

        return param

    def add_adaptive_gate(
        self,
        gate_template: Dict[str, Any],
        optimal_param: Optional[float] = None
    ) -> Tuple[Optional[Parameter], QuantumCircuit]:
        """
        Add a gate from the adaptive pool with optional optimal parameter.

        This is the core method for adaptive circuit construction. It adds
        a gate from the pool and optionally sets its parameter to an
        analytically computed optimal value.

        Args:
            gate_template: Gate template from the adaptive pool
            optimal_param: Pre-computed optimal parameter value

        Returns:
            Tuple of (parameter, updated circuit)
        """
        gate_type = gate_template['type']
        qubits = gate_template['qubits']

        param = None

        if gate_template['parameterized']:
            param = self.gate_set.create_parameter()

            if gate_type in ['rx', 'ry', 'rz']:
                getattr(self.circuit, gate_type)(param, qubits[0])
            elif gate_type in ['crx', 'cry', 'crz']:
                getattr(self.circuit, gate_type)(param, qubits[0], qubits[1])

            self.parameters.append(param)
        else:
            if gate_type == 'cx':
                self.circuit.cx(qubits[0], qubits[1])
            elif gate_type == 'cz':
                self.circuit.cz(qubits[0], qubits[1])

        return param, self.circuit

    def get_circuit(self) -> QuantumCircuit:
        """
        Get the current quantum circuit.

        Returns:
            The quantum circuit
        """
        return self.circuit

    def get_parameters(self) -> List[Parameter]:
        """
        Get all trainable parameters.

        Returns:
            List of parameters
        """
        return self.parameters

    def get_n_parameters(self) -> int:
        """
        Get the number of trainable parameters.

        Returns:
            Number of parameters
        """
        return len(self.parameters)

    def get_circuit_depth(self) -> int:
        """
        Get the current circuit depth.

        Returns:
            Circuit depth
        """
        return self.circuit.depth()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the circuit structure.

        Returns:
            Dictionary containing circuit information
        """
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_parameters': len(self.parameters),
            'circuit_depth': self.circuit.depth(),
            'gate_counts': self.gate_set.get_gate_count(),
            'layer_info': self.layer_info
        }

    def copy(self) -> 'AdaptiveCircuitBuilder':
        """
        Create a deep copy of the builder.

        Returns:
            A new AdaptiveCircuitBuilder with the same state
        """
        new_builder = AdaptiveCircuitBuilder(
            self.n_qubits,
            self.n_classes,
            self.encoding_type,
            self.measurement_type
        )
        new_builder.circuit = self.circuit.copy()
        new_builder.parameters = self.parameters.copy()
        new_builder.n_layers = self.n_layers
        new_builder.layer_info = deepcopy(self.layer_info)
        new_builder.gate_set = deepcopy(self.gate_set)

        return new_builder

    def reset(self) -> None:
        """Reset the builder to initial state."""
        self._initialize_circuit()


def build_standard_circuit(
    n_qubits: int,
    n_layers: int,
    encoding_type: str = 'angle',
    entanglement: str = 'linear'
) -> Tuple[QuantumCircuit, List[Parameter], ParameterVector]:
    """
    Build a standard variational quantum circuit (for comparison).

    This creates a traditional VQC with fixed structure for benchmarking
    against the adaptive approach.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        encoding_type: Data encoding type
        entanglement: Entanglement pattern

    Returns:
        Tuple of (circuit, variational_parameters, data_parameters)
    """
    builder = AdaptiveCircuitBuilder(
        n_qubits,
        encoding_type=encoding_type
    )

    data_params = builder.add_encoding_layer()

    all_params = []
    for _ in range(n_layers):
        params = builder.add_variational_layer(
            layer_type='standard',
            entanglement=entanglement
        )
        all_params.extend(params)

    return builder.get_circuit(), all_params, data_params
