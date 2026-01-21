"""
Quantum Gate Set for Adaptive QNN
==================================

This module defines the parameterized quantum gates used in the adaptive circuit construction.
The gate set is designed to provide universal quantum computation while maintaining
analytical tractability for parameter estimation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from typing import List, Tuple, Optional, Dict, Any


class QuantumGateSet:
    """
    A collection of quantum gates for adaptive QNN construction.

    This class provides methods to add various parameterized gates to quantum circuits,
    with support for both single-qubit rotations and multi-qubit entangling gates.

    The gate set is chosen to:
    1. Enable universal quantum computation
    2. Allow analytical parameter estimation via Fourier analysis
    3. Minimize circuit depth for NISQ devices
    """

    # Gate types for adaptive construction
    SINGLE_QUBIT_GATES = ['rx', 'ry', 'rz']
    TWO_QUBIT_GATES = ['cx', 'cz', 'crx', 'cry', 'crz']
    ENTANGLING_PATTERNS = ['linear', 'circular', 'full', 'alternating']

    def __init__(self, n_qubits: int):
        """
        Initialize the quantum gate set.

        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
        self.parameter_count = 0
        self.gate_history: List[Dict[str, Any]] = []

    def create_parameter(self, name: Optional[str] = None) -> Parameter:
        """
        Create a new parameter for a parameterized gate.

        Args:
            name: Optional name for the parameter

        Returns:
            A Qiskit Parameter object
        """
        if name is None:
            name = f"θ_{self.parameter_count}"
        self.parameter_count += 1
        return Parameter(name)

    def add_rotation_layer(
        self,
        circuit: QuantumCircuit,
        gate_type: str = 'ry',
        qubits: Optional[List[int]] = None,
        parameters: Optional[List[Parameter]] = None
    ) -> List[Parameter]:
        """
        Add a layer of single-qubit rotation gates.

        Args:
            circuit: The quantum circuit to modify
            gate_type: Type of rotation gate ('rx', 'ry', 'rz')
            qubits: List of qubits to apply gates (default: all qubits)
            parameters: List of parameters (default: create new ones)

        Returns:
            List of parameters used in this layer
        """
        if qubits is None:
            qubits = list(range(self.n_qubits))

        if parameters is None:
            parameters = [self.create_parameter() for _ in qubits]

        gate_func = getattr(circuit, gate_type)

        for qubit, param in zip(qubits, parameters):
            gate_func(param, qubit)
            self.gate_history.append({
                'type': gate_type,
                'qubit': qubit,
                'parameter': param
            })

        return parameters

    def add_general_rotation(
        self,
        circuit: QuantumCircuit,
        qubit: int,
        params: Optional[Tuple[Parameter, Parameter, Parameter]] = None
    ) -> Tuple[Parameter, Parameter, Parameter]:
        """
        Add a general single-qubit rotation U3(θ, φ, λ).

        This provides the most general single-qubit operation and is useful
        for adaptive circuit construction where flexibility is needed.

        Args:
            circuit: The quantum circuit to modify
            qubit: Target qubit
            params: Tuple of (theta, phi, lambda) parameters

        Returns:
            Tuple of parameters (theta, phi, lambda)
        """
        if params is None:
            theta = self.create_parameter()
            phi = self.create_parameter()
            lam = self.create_parameter()
            params = (theta, phi, lam)
        else:
            theta, phi, lam = params

        circuit.u(theta, phi, lam, qubit)
        self.gate_history.append({
            'type': 'u3',
            'qubit': qubit,
            'parameters': params
        })

        return params

    def add_entangling_layer(
        self,
        circuit: QuantumCircuit,
        pattern: str = 'linear',
        gate_type: str = 'cx'
    ) -> None:
        """
        Add a layer of entangling gates following a specific pattern.

        Patterns:
        - 'linear': Connect adjacent qubits (0-1, 1-2, 2-3, ...)
        - 'circular': Linear + connect last to first
        - 'full': All-to-all connectivity
        - 'alternating': Even-odd pairs, then odd-even pairs

        Args:
            circuit: The quantum circuit to modify
            pattern: Entanglement pattern
            gate_type: Type of two-qubit gate
        """
        if self.n_qubits < 2:
            return

        pairs = self._get_entanglement_pairs(pattern)
        gate_func = getattr(circuit, gate_type)

        for control, target in pairs:
            gate_func(control, target)
            self.gate_history.append({
                'type': gate_type,
                'control': control,
                'target': target
            })

    def add_controlled_rotation(
        self,
        circuit: QuantumCircuit,
        control: int,
        target: int,
        gate_type: str = 'crx',
        parameter: Optional[Parameter] = None
    ) -> Parameter:
        """
        Add a controlled rotation gate.

        Args:
            circuit: The quantum circuit to modify
            control: Control qubit
            target: Target qubit
            gate_type: Type of controlled rotation ('crx', 'cry', 'crz')
            parameter: Optional parameter (default: create new one)

        Returns:
            The parameter used for the gate
        """
        if parameter is None:
            parameter = self.create_parameter()

        gate_func = getattr(circuit, gate_type)
        gate_func(parameter, control, target)

        self.gate_history.append({
            'type': gate_type,
            'control': control,
            'target': target,
            'parameter': parameter
        })

        return parameter

    def add_variational_block(
        self,
        circuit: QuantumCircuit,
        rotation_gates: List[str] = ['ry', 'rz'],
        entanglement: str = 'linear'
    ) -> List[Parameter]:
        """
        Add a standard variational block (rotation layer + entangling layer).

        This is a common building block in variational quantum circuits,
        consisting of a layer of parameterized rotations followed by
        an entangling layer.

        Args:
            circuit: The quantum circuit to modify
            rotation_gates: List of rotation gates to apply
            entanglement: Entanglement pattern

        Returns:
            List of parameters created for this block
        """
        parameters = []

        for gate_type in rotation_gates:
            params = self.add_rotation_layer(circuit, gate_type)
            parameters.extend(params)

        self.add_entangling_layer(circuit, pattern=entanglement)

        return parameters

    def _get_entanglement_pairs(self, pattern: str) -> List[Tuple[int, int]]:
        """
        Get qubit pairs for a given entanglement pattern.

        Args:
            pattern: Entanglement pattern name

        Returns:
            List of (control, target) pairs
        """
        pairs = []

        if pattern == 'linear':
            pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]

        elif pattern == 'circular':
            pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]

        elif pattern == 'full':
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    pairs.append((i, j))

        elif pattern == 'alternating':
            # Even-odd pairs
            pairs.extend([(i, i + 1) for i in range(0, self.n_qubits - 1, 2)])
            # Odd-even pairs
            pairs.extend([(i, i + 1) for i in range(1, self.n_qubits - 1, 2)])

        return pairs

    def get_gate_count(self) -> Dict[str, int]:
        """
        Get the count of each gate type used.

        Returns:
            Dictionary mapping gate types to counts
        """
        counts = {}
        for gate in self.gate_history:
            gate_type = gate['type']
            counts[gate_type] = counts.get(gate_type, 0) + 1
        return counts

    def reset(self) -> None:
        """Reset the gate set state."""
        self.parameter_count = 0
        self.gate_history.clear()


def create_adaptive_gate_pool(n_qubits: int, reduced: bool = True) -> List[Dict[str, Any]]:
    """
    Create a pool of gate templates for adaptive circuit construction.

    This function generates a pool of possible gates that can be added
    during the adaptive construction process. Each gate template includes
    the gate type, target qubits, and whether it introduces a new parameter.

    Args:
        n_qubits: Number of qubits
        reduced: If True, use reduced gate pool for faster training

    Returns:
        List of gate templates
    """
    pool = []

    # Single-qubit rotation gates (always include)
    for gate_type in ['ry']:  # Ry is most useful for amplitude manipulation
        for qubit in range(n_qubits):
            pool.append({
                'type': gate_type,
                'qubits': [qubit],
                'parameterized': True,
                'n_params': 1
            })

    if not reduced:
        # Add rx and rz for full expressivity
        for gate_type in ['rx', 'rz']:
            for qubit in range(n_qubits):
                pool.append({
                    'type': gate_type,
                    'qubits': [qubit],
                    'parameterized': True,
                    'n_params': 1
                })

    # Two-qubit entangling gates - only linear connectivity for speed
    for i in range(n_qubits - 1):
        pool.append({
            'type': 'cx',
            'qubits': [i, i + 1],
            'parameterized': False,
            'n_params': 0
        })
        pool.append({
            'type': 'cx',
            'qubits': [i + 1, i],
            'parameterized': False,
            'n_params': 0
        })

    # Controlled rotation gates - only linear connectivity
    for gate_type in ['cry']:  # CRY is most useful
        for i in range(n_qubits - 1):
            pool.append({
                'type': gate_type,
                'qubits': [i, i + 1],
                'parameterized': True,
                'n_params': 1
            })
            pool.append({
                'type': gate_type,
                'qubits': [i + 1, i],
                'parameterized': True,
                'n_params': 1
            })

    if not reduced:
        # Full connectivity for controlled rotations
        for gate_type in ['crx', 'crz']:
            for control in range(n_qubits):
                for target in range(n_qubits):
                    if control != target:
                        pool.append({
                            'type': gate_type,
                            'qubits': [control, target],
                            'parameterized': True,
                            'n_params': 1
                        })

    return pool
