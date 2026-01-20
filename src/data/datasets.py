"""
Dataset Loaders for Quantum Neural Networks
=============================================

Provides standard datasets prepared for quantum machine learning.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import load_iris, make_moons, make_circles


def load_iris_quantum(
    n_qubits: int = 4,
    binary: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Iris dataset prepared for quantum processing.

    Args:
        n_qubits: Number of qubits (features will be reduced to this)
        binary: If True, use only 2 classes
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    iris = load_iris()
    X, y = iris.data, iris.target

    # Binary classification: class 0 vs class 1
    if binary:
        mask = y < 2
        X, y = X[mask], y[mask]

    # Reduce features if needed
    if X.shape[1] > n_qubits:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_qubits)
        X = pca.fit_transform(X)

    # Scale to [0, π] for angle encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_moons_quantum(
    n_samples: int = 200,
    n_qubits: int = 2,
    noise: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load make_moons dataset for quantum processing.

    Two interleaving half circles - a classic nonlinear classification problem.

    Args:
        n_samples: Total number of samples
        n_qubits: Number of qubits (will pad features if needed)
        noise: Noise level in the data
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # Scale to [0, π]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    # Pad with zeros if more qubits needed
    if X.shape[1] < n_qubits:
        padding = np.zeros((X.shape[0], n_qubits - X.shape[1]))
        X = np.hstack([X, padding])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_circles_quantum(
    n_samples: int = 200,
    n_qubits: int = 2,
    noise: float = 0.1,
    factor: float = 0.5,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load make_circles dataset for quantum processing.

    Two concentric circles - requires nonlinear decision boundary.

    Args:
        n_samples: Total number of samples
        n_qubits: Number of qubits
        noise: Noise level
        factor: Scale factor between circles
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state
    )

    # Scale to [0, π]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    # Pad if needed
    if X.shape[1] < n_qubits:
        padding = np.zeros((X.shape[0], n_qubits - X.shape[1]))
        X = np.hstack([X, padding])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def generate_quantum_data(
    n_samples: int = 200,
    n_features: int = 4,
    n_classes: int = 2,
    pattern: str = 'xor',
    noise: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum-friendly data.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        pattern: Data pattern ('xor', 'parity', 'random', 'linear')
        noise: Noise level
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    np.random.seed(random_state)

    # Generate features uniformly in [0, π]
    X = np.random.uniform(0, np.pi, size=(n_samples, n_features))

    if pattern == 'xor':
        # XOR pattern: class based on product of signs
        # Convert to [-1, 1] for XOR computation
        X_centered = X - np.pi/2
        y = (np.sign(X_centered[:, 0]) * np.sign(X_centered[:, 1]) > 0).astype(int)

    elif pattern == 'parity':
        # Parity: class based on sum of binary features
        X_binary = (X > np.pi/2).astype(int)
        y = np.sum(X_binary, axis=1) % n_classes

    elif pattern == 'linear':
        # Linear separation
        threshold = np.sum(X, axis=1) / n_features
        y = (threshold > np.pi/2).astype(int)

    else:  # random
        y = np.random.randint(0, n_classes, n_samples)

    # Add noise
    X = X + np.random.normal(0, noise, X.shape)
    X = np.clip(X, 0, np.pi)  # Keep in valid range

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_breast_cancer_quantum(
    n_qubits: int = 4,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Breast Cancer dataset for quantum processing.

    A classic binary classification dataset with 30 features.

    Args:
        n_qubits: Number of qubits
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    data = load_breast_cancer()
    X, y = data.data, data.target

    # Scale first
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensions
    pca = PCA(n_components=n_qubits)
    X_reduced = pca.fit_transform(X_scaled)

    # Scale to [0, π] for angle encoding
    scaler2 = MinMaxScaler(feature_range=(0, np.pi))
    X_final = scaler2.fit_transform(X_reduced)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_digits_quantum(
    n_qubits: int = 8,
    n_classes: int = 2,
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load handwritten digits dataset for quantum processing.

    Args:
        n_qubits: Number of qubits
        n_classes: Number of digit classes to use
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    digits = load_digits()
    X, y = digits.data, digits.target

    # Filter to n_classes
    mask = y < n_classes
    X, y = X[mask], y[mask]

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensions
    pca = PCA(n_components=n_qubits)
    X_reduced = pca.fit_transform(X_scaled)

    # Scale to [0, π]
    scaler2 = MinMaxScaler(feature_range=(0, np.pi))
    X_final = scaler2.fit_transform(X_reduced)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
