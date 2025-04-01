import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

def generate_synthetic_regression_data(n_samples=1000, n_features=50, noise=0.1, random_seed=42):
    np.random.seed(random_seed)
    X = np.random.rand(n_samples, n_features)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X = (X - X_mean) / X_std

    y_true = np.sin(2 * np.pi * X[:, 0])
    y_noisy = y_true + noise * np.random.randn(n_samples)

    y_noisy = y_noisy.reshape(-1, 1)

    return X, y_noisy