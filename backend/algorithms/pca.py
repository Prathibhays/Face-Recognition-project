import numpy as np


class PCAEngine:
    # Educational PCA implementation using eigen-decomposition.

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")

        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real(eigenvalues[order])
        eigenvectors = np.real(eigenvectors[:, order])
        eigenvalues = np.maximum(eigenvalues, 0)

        total = np.sum(eigenvalues)
        ratios = eigenvalues / total if total else np.zeros_like(eigenvalues)

        k = min(int(self.n_components), len(eigenvalues), n_samples, n_features)

        self.eigenvalues_ = eigenvalues[:k]
        self.components_ = eigenvectors[:, :k].T
        self.explained_variance_ratio_ = ratios[:k]
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAEngine has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        return Z @ self.components_ + self.mean_
