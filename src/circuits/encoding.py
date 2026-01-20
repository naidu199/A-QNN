"""
Data Encoding Strategies for Quantum Neural Networks
=====================================================

This module provides various data encoding strategies for mapping classical
data into quantum states. The choice of encoding significantly impacts the
expressibility and trainability of QNNs.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from typing import List, Tuple, Optional, Union
from scipy.special import softmax


class DataEncoder:
    """
    Encodes classical data into quantum states.

    This class provides multiple encoding strategies, each with different
    properties regarding expressibility, trainability, and resource requirements.

    Encoding Strategies:
    - Angle Encoding: Maps features to rotation angles
    - Amplitude Encoding: Maps features to state amplitudes
    - Basis Encoding: Maps binary features to computational basis states
    - IQP Encoding: Instantaneous Quantum Polynomial encoding
    - Dense Angle Encoding: Multiple features per qubit using Bloch sphere
    """

    SUPPORTED_ENCODINGS = [
        'angle', 'amplitude', 'basis', 'iqp',
        'dense_angle', 'hamiltonian'
    ]

    def __init__(
        self,
        n_qubits: int,
        encoding_type: str = 'angle',
        reps: int = 1,
        normalize: bool = True
    ):
        """
        Initialize the data encoder.

        Args:
            n_qubits: Number of qubits available
            encoding_type: Type of encoding to use
            reps: Number of encoding repetitions (for re-uploading)
            normalize: Whether to normalize input data
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.reps = reps
        self.normalize = normalize

        self._validate_encoding()

    def _validate_encoding(self) -> None:
        """Validate the encoding type."""
        if self.encoding_type not in self.SUPPORTED_ENCODINGS:
            raise ValueError(
                f"Unknown encoding type: {self.encoding_type}. "
                f"Supported: {self.SUPPORTED_ENCODINGS}"
            )

    def get_n_features(self) -> int:
        """
        Get the number of features this encoding expects.

        Returns:
            Number of input features
        """
        if self.encoding_type == 'angle':
            return self.n_qubits
        elif self.encoding_type == 'dense_angle':
            return 2 * self.n_qubits  # θ and φ per qubit
        elif self.encoding_type == 'amplitude':
            return 2 ** self.n_qubits
        elif self.encoding_type == 'basis':
            return self.n_qubits  # Binary features
        elif self.encoding_type == 'iqp':
            return self.n_qubits
        elif self.encoding_type == 'hamiltonian':
            return self.n_qubits
        return self.n_qubits

    def encode(
        self,
        circuit: QuantumCircuit,
        data: Union[np.ndarray, ParameterVector],
        start_qubit: int = 0
    ) -> QuantumCircuit:
        """
        Encode classical data into the quantum circuit.

        Args:
            circuit: Quantum circuit to add encoding to
            data: Classical data (array or parameter vector)
            start_qubit: First qubit to use for encoding

        Returns:
            Modified quantum circuit
        """
        if self.encoding_type == 'angle':
            return self._angle_encoding(circuit, data, start_qubit)
        elif self.encoding_type == 'dense_angle':
            return self._dense_angle_encoding(circuit, data, start_qubit)
        elif self.encoding_type == 'amplitude':
            return self._amplitude_encoding(circuit, data, start_qubit)
        elif self.encoding_type == 'basis':
            return self._basis_encoding(circuit, data, start_qubit)
        elif self.encoding_type == 'iqp':
            return self._iqp_encoding(circuit, data, start_qubit)
        elif self.encoding_type == 'hamiltonian':
            return self._hamiltonian_encoding(circuit, data, start_qubit)

        return circuit

    def _angle_encoding(
        self,
        circuit: QuantumCircuit,
        data: Union[np.ndarray, ParameterVector],
        start_qubit: int
    ) -> QuantumCircuit:
        """
        Angle encoding: Each feature maps to a rotation angle.

        For each qubit i: |0⟩ → RY(x_i)|0⟩
        """
        # Initialize with Hadamard for superposition
        for i in range(self.n_qubits):
            circuit.h(start_qubit + i)

        # Apply rotations for each repetition
        for rep in range(self.reps):
            for i in range(min(len(data), self.n_qubits)):
                # Scale data to [0, 2π] range
                if isinstance(data, np.ndarray):
                    angle = np.pi * data[i]
                else:
                    angle = data[i]
                circuit.ry(angle, start_qubit + i)

                # Add entanglement between reps
                if rep < self.reps - 1 and i < self.n_qubits - 1:
                    circuit.cx(start_qubit + i, start_qubit + i + 1)

        return circuit

    def _dense_angle_encoding(
        self,
        circuit: QuantumCircuit,
        data: Union[np.ndarray, ParameterVector],
        start_qubit: int
    ) -> QuantumCircuit:
        """
        Dense angle encoding: Use full Bloch sphere (θ, φ) per qubit.

        Encodes 2 features per qubit using U3(θ, φ, 0) gates.
        """
        for i in range(self.n_qubits):
            idx = 2 * i
            if idx < len(data):
                theta = data[idx] if isinstance(data, ParameterVector) else np.pi * data[idx]
                phi = data[idx + 1] if idx + 1 < len(data) else 0
                if isinstance(data, np.ndarray) and idx + 1 < len(data):
                    phi = np.pi * data[idx + 1]
                circuit.u(theta, phi, 0, start_qubit + i)

        return circuit

    def _amplitude_encoding(
        self,
        circuit: QuantumCircuit,
        data: Union[np.ndarray, ParameterVector],
        start_qubit: int
    ) -> QuantumCircuit:
        """
        Amplitude encoding: Map data to state amplitudes.

        Note: True amplitude encoding requires state preparation circuits.
        This is an approximation using parameterized rotations.
        """
        if isinstance(data, np.ndarray):
            # Normalize for amplitude encoding
            if self.normalize:
                norm = np.linalg.norm(data)
                if norm > 0:
                    data = data / norm

            # Use approximate preparation via rotations
            # This is a simplified version - full amplitude encoding
            # requires more complex state preparation

        # Apply initial Hadamards
        for i in range(self.n_qubits):
            circuit.h(start_qubit + i)

        # Apply parameterized rotations
        for i in range(min(len(data), self.n_qubits)):
            if isinstance(data, np.ndarray):
                angle = 2 * np.arcsin(np.clip(data[i], -1, 1))
            else:
                angle = data[i]
            circuit.ry(angle, start_qubit + i)

        return circuit

    def _basis_encoding(
        self,
        circuit: QuantumCircuit,
        data: Union[np.ndarray, ParameterVector],
        start_qubit: int
    ) -> QuantumCircuit:
        """
        Basis encoding: Map binary features to computational basis.

        For binary data x_i ∈ {0, 1}: Apply X gate if x_i = 1
        """
        if isinstance(data, ParameterVector):
            raise ValueError("Basis encoding requires concrete binary values")

        for i in range(min(len(data), self.n_qubits)):
            if data[i] > 0.5:  # Threshold for binary
                circuit.x(start_qubit + i)

        return circuit

    def _iqp_encoding(
        self,
        circuit: QuantumCircuit,
        data: Union[np.ndarray, ParameterVector],
        start_qubit: int
    ) -> QuantumCircuit:
        """
        IQP (Instantaneous Quantum Polynomial) encoding.

        Creates entanglement between features through ZZ interactions,
        which can capture nonlinear feature relationships.
        """
        n = min(len(data), self.n_qubits)

        # First Hadamard layer
        for i in range(n):
            circuit.h(start_qubit + i)

        # Single qubit Z rotations
        for i in range(n):
            if isinstance(data, np.ndarray):
                angle = np.pi * data[i]
            else:
                angle = data[i]
            circuit.rz(angle, start_qubit + i)

        # ZZ interactions (feature products)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if isinstance(data, np.ndarray):
                    angle = np.pi * data[i] * data[j]
                else:
                    angle = data[i] * data[j]
                # ZZ interaction via CNOT-RZ-CNOT
                circuit.cx(start_qubit + i, start_qubit + j)
                circuit.rz(angle, start_qubit + j)
                circuit.cx(start_qubit + i, start_qubit + j)

        # Second Hadamard layer
        for i in range(n):
            circuit.h(start_qubit + i)

        return circuit

    def _hamiltonian_encoding(
        self,
        circuit: QuantumCircuit,
        data: Union[np.ndarray, ParameterVector],
        start_qubit: int
    ) -> QuantumCircuit:
        """
        Hamiltonian evolution encoding.

        Encodes data as Hamiltonian parameters: exp(-i H(x) t)
        """
        n = min(len(data), self.n_qubits)

        # Initial superposition
        for i in range(n):
            circuit.h(start_qubit + i)

        # Hamiltonian evolution approximation
        for i in range(n):
            if isinstance(data, np.ndarray):
                angle = data[i]
            else:
                angle = data[i]

            # X component
            circuit.rx(angle, start_qubit + i)

            # Z component
            circuit.rz(angle, start_qubit + i)

            # ZZ interaction
            if i < n - 1:
                circuit.cx(start_qubit + i, start_qubit + i + 1)
                circuit.rz(angle, start_qubit + i + 1)
                circuit.cx(start_qubit + i, start_qubit + i + 1)

        return circuit

    def preprocess_data(
        self,
        X: np.ndarray,
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Preprocess classical data before encoding.

        Args:
            X: Input data array (n_samples, n_features)
            method: Preprocessing method
                - 'minmax': Scale to [0, 1]
                - 'standard': Zero mean, unit variance
                - 'arctan': Apply arctan scaling

        Returns:
            Preprocessed data
        """
        if method == 'minmax':
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1  # Avoid division by zero
            return (X - X_min) / X_range

        elif method == 'standard':
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1
            return (X - mean) / std

        elif method == 'arctan':
            return (np.arctan(X) + np.pi/2) / np.pi

        return X

    def create_encoding_circuit(
        self,
        n_features: Optional[int] = None
    ) -> Tuple[QuantumCircuit, ParameterVector]:
        """
        Create a standalone encoding circuit with parameter placeholders.

        Args:
            n_features: Number of features (default: based on encoding type)

        Returns:
            Tuple of (circuit, data_parameters)
        """
        if n_features is None:
            n_features = self.get_n_features()

        circuit = QuantumCircuit(self.n_qubits, name=f'{self.encoding_type}_encoding')
        data_params = ParameterVector('x', n_features)

        self.encode(circuit, data_params)

        return circuit, data_params


def select_optimal_encoding(
    n_qubits: int,
    n_features: int,
    task_type: str = 'classification'
) -> str:
    """
    Select optimal encoding strategy based on problem characteristics.

    Args:
        n_qubits: Available qubits
        n_features: Number of input features
        task_type: 'classification' or 'regression'

    Returns:
        Recommended encoding type
    """
    if n_features <= n_qubits:
        # Few features: angle encoding is efficient
        return 'angle'
    elif n_features <= 2 * n_qubits:
        # Medium features: use dense encoding
        return 'dense_angle'
    elif n_features == 2 ** n_qubits:
        # Exact power of 2: amplitude encoding possible
        return 'amplitude'
    else:
        # Many features: IQP can capture interactions
        return 'iqp'
