"""
Data Preprocessing for Quantum Neural Networks
================================================

This module provides preprocessing utilities specifically designed for
quantum machine learning tasks.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Preprocessor for quantum machine learning data.

    Handles common preprocessing tasks:
    - Feature scaling (important for angle encoding)
    - Dimensionality reduction (to match qubit count)
    - Label encoding
    - Train/test splitting

    Example:
        >>> preprocessor = DataPreprocessor(n_qubits=4)
        >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
    """

    def __init__(
        self,
        n_qubits: int,
        scaling: str = 'minmax',
        reduce_dim: bool = True,
        target_range: Tuple[float, float] = (0, np.pi)
    ):
        """
        Initialize the preprocessor.

        Args:
            n_qubits: Number of qubits (determines max features)
            scaling: Scaling method ('minmax', 'standard', 'none')
            reduce_dim: Whether to reduce dimensionality to n_qubits
            target_range: Target range for scaled features
        """
        self.n_qubits = n_qubits
        self.scaling = scaling
        self.reduce_dim = reduce_dim
        self.target_range = target_range

        # Initialize components
        self.scaler = None
        self.pca = None
        self.label_encoder = LabelEncoder()

        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.

        Args:
            X: Features, shape (n_samples, n_features)
            y: Labels, shape (n_samples,)

        Returns:
            self
        """
        # Fit scaler
        if self.scaling == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.target_range)
            self.scaler.fit(X)
        elif self.scaling == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        # Fit PCA if needed
        if self.reduce_dim and X.shape[1] > self.n_qubits:
            n_components = min(self.n_qubits, X.shape[1], X.shape[0])
            self.pca = PCA(n_components=n_components)

            # Transform for PCA fitting
            X_scaled = self.scaler.transform(X) if self.scaler else X
            self.pca.fit(X_scaled)

        # Fit label encoder
        self.label_encoder.fit(y)

        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Features
            y: Optional labels

        Returns:
            Transformed X, and optionally transformed y
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted first")

        # Scale
        if self.scaler:
            X_transformed = self.scaler.transform(X)
        else:
            X_transformed = X.copy()

        # Reduce dimensionality
        if self.pca is not None:
            X_transformed = self.pca.transform(X_transformed)

            # Re-scale after PCA
            if self.scaling == 'minmax':
                X_min = X_transformed.min(axis=0)
                X_max = X_transformed.max(axis=0)
                X_range = X_max - X_min
                X_range[X_range == 0] = 1
                X_transformed = (X_transformed - X_min) / X_range
                X_transformed = X_transformed * (self.target_range[1] - self.target_range[0])
                X_transformed = X_transformed + self.target_range[0]

        if y is not None:
            y_transformed = self.label_encoder.transform(y)
            return X_transformed, y_transformed

        return X_transformed

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data with train/test split.

        Args:
            X: Features
            y: Labels
            test_size: Fraction for test set
            random_state: Random seed

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Split first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Fit on training data
        self.fit(X_train, y_train)

        # Transform both sets
        X_train_transformed, y_train_transformed = self.transform(X_train, y_train)
        X_test_transformed, y_test_transformed = self.transform(X_test, y_test)

        return (X_train_transformed, X_test_transformed,
                y_train_transformed, y_test_transformed)

    def get_n_features(self) -> int:
        """Get number of features after preprocessing."""
        if self.pca is not None:
            return self.pca.n_components_
        return self.n_qubits

    def get_n_classes(self) -> int:
        """Get number of classes."""
        return len(self.label_encoder.classes_)

    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original."""
        return self.label_encoder.inverse_transform(y)


def prepare_quantum_data(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    test_size: float = 0.2,
    encoding: str = 'angle',
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Convenience function to prepare data for quantum processing.

    Args:
        X: Raw features
        y: Raw labels
        n_qubits: Number of qubits
        test_size: Test set fraction
        encoding: Encoding type ('angle', 'amplitude')
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test, info_dict
    """
    # Determine target range based on encoding
    if encoding == 'angle':
        target_range = (0, np.pi)
    elif encoding == 'amplitude':
        target_range = (-1, 1)
    else:
        target_range = (0, 1)

    preprocessor = DataPreprocessor(
        n_qubits=n_qubits,
        scaling='minmax',
        reduce_dim=True,
        target_range=target_range
    )

    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        X, y, test_size=test_size, random_state=random_state
    )

    info = {
        'n_features': preprocessor.get_n_features(),
        'n_classes': preprocessor.get_n_classes(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'encoding': encoding,
        'target_range': target_range
    }

    return X_train, X_test, y_train, y_test, info


def augment_quantum_data(
    X: np.ndarray,
    y: np.ndarray,
    n_augmented: int,
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment quantum data with small perturbations.

    Useful for increasing training data size while maintaining
    label consistency.

    Args:
        X: Features
        y: Labels
        n_augmented: Number of augmented samples per original
        noise_level: Standard deviation of Gaussian noise
        random_state: Random seed

    Returns:
        Augmented X and y
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_aug = [X]
    y_aug = [y]

    for _ in range(n_augmented):
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)
