from optimizers import smoothed, compute_diameter, compute_mse_lipschitz_constant
from testFunctions import TestFunctions
import numpy as np
import matplotlib.pyplot as plt
from datasets import generate_synthetic_regression_data
from ELM import ELM
from copy import deepcopy
import time
from scipy.stats import linregress
import pandas as pd

# ===== Test Functions =====
theta_init = np.array([-0.012, 0.052])
max_iters= 10000

def smoothed_optimizer_wrapper(f):
    _, history = smoothed(
        f = f,
        theta_init = theta_init,
        epochs= max_iters,
        M=1.0,
        mu = None,
        nu = 1e-3,
    )
    return history

TestFunctions().plot_contours(smoothed_optimizer_wrapper)

# ===== Synthetic Data =====
X_syn, y_syn = generate_synthetic_regression_data(n_samples=1000, n_features=50, noise=0.1, random_seed=42)
print(f"X shape: {X_syn.shape}, y shape: {y_syn.shape}")
print(f"X range: {X_syn.min():.2f} to {X_syn.max():.2f}")
print(f"y range: {y_syn.min():.2f} to {y_syn.max():.2f}")

# --- experiments with mu
mu_strategies = {
    'theoretical mu': None,
    'fixed_1e-4': 1e-4,
    'fixed_1e-3': 1e-3,
    'fixed_1e-2': 1e-2
}

results = {}
for mu_key, mu in mu_strategies.items():
    print(f"\nRunning experiment for: {mu_key}")

    model = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
                l1_lambda=0.001, activation='relu', seed=1710)

    model.forward(X_syn)
    H = model.hidden_layer_output
    D1 = compute_diameter(X_syn)
    hidden_repr = model.activation(X_syn @ model.weights_input_hidden + model.bias_hidden)
    D2 = compute_diameter(hidden_repr)
    M = compute_mse_lipschitz_constant(model, X_syn)

    start_time = time.time()
    history = model.train(
        X_syn, y_syn,
        D1=D1,
        D2=D2,
        optimizer=smoothed,
        epochs=9000,
        M=M,
        mu=mu,
        momentum_init=0.9,
        gradient_norm_threshold=1e-6,
        verbose = True
    )
    execution_time = time.time() - start_time

    results[mu_key] = {
        'loss_history': history["losses"],
        'grad_norms': history["grad_norms"],
        'execution_time': execution_time
    }

    print(f"Final Loss: {history['losses'][-1]:.6f}, Execution Time: {execution_time:.2f} sec")

epochs = 9000
log_gap_data = {}

for mu_key, history in results.items():
    loss_history = np.array(history['loss_history'])
    min_loss = np.min(loss_history)  # f(x*)
    relative_gap = (loss_history - min_loss) / abs(min_loss)
    log_gap_data[mu_key] = relative_gap

# relative gap plot using log-scale
plt.figure(figsize=(10, 6))
for mu_key, gap_values in log_gap_data.items():
    plt.plot(range(len(gap_values)), gap_values, label=mu_key, alpha=0.8)

plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Relative Gap (log scale)")
plt.title("Convergence Comparison for Different Values of Mu")
plt.legend()
plt.grid(True)
plt.savefig("convergence_comparison_different_mu.png", dpi=300)

# Analysis of the empirical slope
convergence_rates = {}
for mu_key, gap_values in log_gap_data.items():
    valid_idx = np.where(gap_values > 1e-8)[0]  # avoid log(0)
    if len(valid_idx) < 10:
        continue

    x_vals = np.log(valid_idx + 1)
    y_vals = np.log(gap_values[valid_idx])

    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    convergence_rates[mu_key] = slope

convergence_df = pd.DataFrame.from_dict(convergence_rates, orient="index", columns=["Slope (Rate)"])
convergence_df = convergence_df.sort_values(by="Slope (Rate)", ascending=True)
print(convergence_df)


convergence_slopes = {}
for mu_key, gap_values in log_gap_data.items():
    valid_idx = np.where(gap_values > 1e-8)[0]
    if len(valid_idx) < 10:
        continue

    x_vals = np.log(valid_idx + 1)  # Index
    y_vals = np.log(gap_values[valid_idx])  # Gap

    # Stages:
    n = len(x_vals)
    segments = {
        "early": (0, n // 3),
        "mid": (n // 3, 2 * n // 3),
        "late": (2 * n // 3, n - 1)
    }

    slopes = {}
    for phase, (start, end) in segments.items():
        if end - start > 5:
            slope, _, _, _, _ = linregress(x_vals[start:end], y_vals[start:end])
            slopes[phase] = slope

    convergence_slopes[mu_key] = slopes
    plt.plot(np.log(valid_idx + 1), np.log(gap_values[valid_idx]), label=mu_key, alpha=0.8)

plt.xlabel(r"$\log(k)$ (log epoch)")
plt.ylabel(r"$\log(\text{gap})$ (log relative gap)")
plt.title("Convergence analysis (log-log)")
plt.legend()
plt.grid(True)
plt.savefig("convergence_analysis_log_log.png", dpi=300)

convergence_slopes_df = pd.DataFrame(convergence_slopes).T
print(convergence_slopes_df)

# --- experiments with lambda
from utils import grid_search_smoothed_lambda
best_lambda, search_results = grid_search_smoothed_lambda(X_syn, y_syn)
print(f"Best λ = {best_lambda[0]}, Final Loss = {best_lambda[1]:.6f}")

plt.figure(figsize=(10, 5))

for lambda_key, result in search_results.items():
    losses = result['loss_history']
    final_loss = losses[-1]
    relative_gap = [(loss - final_loss) / final_loss for loss in losses]  # Compute relative gap

    plt.semilogy(range(1, len(relative_gap) + 1), relative_gap, label=lambda_key)

plt.xlabel("Iterations")
plt.ylabel("Relative Gap")
plt.title("Convergence Comparison for Different λ Values")
plt.legend()
plt.grid(True)
plt.savefig("convergence_comparison_lambda_smoothed.png", dpi=300)

# ===== Grid Search finale (per selezionare best params) =====
from utils import grid_search_hyperparams_smoothed

param_grid = {
    'momentum_init': [0.5, 0.8, 0.9],
    'mu': [1e-4, 1e-3, 1e-2, None],
    'nu': [1e-4, 1e-3, None]
}

best_hyperparams, best_loss, search_results = grid_search_hyperparams_smoothed(X_syn, y_syn, param_grid, epochs=1000)

print("BEST PARAMETERS COMBINATION (smoothed):")
print(f"{best_hyperparams}, Final Loss = {best_loss:.6f}")

# ===== California Housing =====
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_california_housing_data(random_state=42, sample_fraction=0.05):
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    np.random.seed(random_state) # random sample
    sample_size = int(sample_fraction * X.shape[0])
    indices = np.random.choice(X.shape[0], sample_size, replace=False)

    return X[indices], y[indices]

X_cal, y_cal = load_california_housing_data()
print(f"X_cal shape: {X_cal.shape}, y_cal shape: {y_cal.shape}")

results = {}

for mu_key, mu in mu_strategies.items():
    print(f"\nRunning experiment for: {mu_key}")

    model = ELM(input_size=X_cal.shape[1], hidden_size=200, output_size=1,
                l1_lambda=0.001, activation='relu', seed=1710)

    model.forward(X_cal)
    H = model.hidden_layer_output
    D1 = compute_diameter(X_cal)
    hidden_repr = model.activation(X_cal @ model.weights_input_hidden + model.bias_hidden)
    D2 = compute_diameter(hidden_repr)
    M = compute_mse_lipschitz_constant(model, X_cal)

    start_time = time.time()
    history = model.train(
        X_cal, y_cal,
        A_norm=1.0,
        D1=D1,
        D2=D2,
        optimizer=smoothed,
        epochs=8000,
        M=M,
        mu=mu,
        momentum_init=0.7,
        gradient_norm_threshold=1e-6
    )
    execution_time = time.time() - start_time

    results[mu_key] = {
        'loss_history': history["losses"],
        'grad_norms': history["grad_norms"],
        'execution_time': execution_time
    }

    print(f"Final Loss: {history['losses'][-1]:.6f}, Execution Time: {execution_time:.2f} sec")

epochs = 5000
log_gap_data = {}

for mu_key, history in results.items():
    loss_history = np.array(history['loss_history'])
    min_loss = np.min(loss_history)  # f(x*)
    relative_gap = (loss_history - min_loss) / abs(min_loss)

    log_gap_data[mu_key] = relative_gap

plt.figure(figsize=(10, 6))
for mu_key, gap_values in log_gap_data.items():
    plt.plot(range(len(gap_values)), gap_values, label=mu_key, alpha=0.8)

plt.yscale("log")
plt.xlabel("epochs")
plt.ylabel("Relative gap (log scale)")
plt.title("Convergence comparison for different values of mu (california)")
plt.legend()
plt.grid(True)
plt.savefig("convergence_comparison_different_mu_california_smoothed.png", dpi=300)

convergence_rates = {}
for mu_key, gap_values in log_gap_data.items():
    valid_idx = np.where(gap_values > 1e-8)[0]
    if len(valid_idx) < 10:
        continue

    x_vals = np.log(valid_idx + 1)
    y_vals = np.log(gap_values[valid_idx])

    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    convergence_rates[mu_key] = slope

convergence_df = pd.DataFrame.from_dict(convergence_rates, orient="index", columns=["Slope (Rate)"])
convergence_df = convergence_df.sort_values(by="Slope (Rate)", ascending=True)
print(convergence_df)

convergence_slopes = {}
for mu_key, gap_values in log_gap_data.items():
    valid_idx = np.where(gap_values > 1e-8)[0]
    if len(valid_idx) < 10:
        continue

    x_vals = np.log(valid_idx + 1)
    y_vals = np.log(gap_values[valid_idx])

    n = len(x_vals)
    segments = {
        "early": (0, n // 3),
        "mid": (n // 3, 2 * n // 3),
        "late": (2 * n // 3, n - 1)
    }

    slopes = {}
    for phase, (start, end) in segments.items():
        if end - start > 5:
            slope, _, _, _, _ = linregress(x_vals[start:end], y_vals[start:end])
            slopes[phase] = slope

    convergence_slopes[mu_key] = slopes
    plt.plot(np.log(valid_idx + 1), np.log(gap_values[valid_idx]), label=mu_key, alpha=0.8)

plt.xlabel(r"$\log(k)$ (log epoch)")
plt.ylabel(r"$\log(\text{gap})$ (log relative gap)")
plt.title("Convergence analysis (log-log) for California Housing")
plt.legend()
plt.grid(True)
plt.savefig("convergence_analysis_log_log_california.png", dpi=300)

convergence_slopes_df = pd.DataFrame(convergence_slopes).T
print(convergence_slopes_df)

# --- experiments with lambda (california)
results = {}

best_lambda, search_results = grid_search_smoothed_lambda(X_cal, y_cal, momentum= 0.8)
print("BEST λ VALUE (california):")
print(f"λ = {best_lambda[0]}, Final Loss = {best_lambda[1]:.6f}")


plt.figure(figsize=(10, 5))

for lambda_key, result in search_results.items():
    losses = result['loss_history']
    final_loss = losses[-1]
    relative_gap = [(loss - final_loss) / final_loss for loss in losses]

    plt.semilogy(range(1, len(relative_gap) + 1), relative_gap, label=lambda_key)

plt.xlabel("Iterations")
plt.ylabel("Relative Gap")
plt.title("Convergence Comparison for Different λ Values")
plt.legend()
plt.grid(True)
plt.savefig("convergence_comparison_lambda_smoothed_california.png", dpi=300)