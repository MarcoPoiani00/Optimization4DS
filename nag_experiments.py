from ELM import ELM
import numpy as np
from datasets import generate_synthetic_regression_data
from optimizers import NAG, optimizer_wrapper_nag
from copy import deepcopy
import matplotlib.pyplot as plt

# ==== Test Functions =====
from testFunctions import TestFunctions

TestFunctions().plot_contours(optimizer_wrapper_nag)

# ===== Synthetic Data =====
X_syn, y_syn = generate_synthetic_regression_data(n_samples=1000, n_features=50, noise=0.1, random_seed=42)
print(f"X shape: {X_syn.shape}, y shape: {y_syn.shape}")
print(f"X range: {X_syn.min():.2f} to {X_syn.max():.2f}")
print(f"y range: {y_syn.min():.2f} to {y_syn.max():.2f}")

model_original = ELM(input_size=X_syn.shape[1], hidden_size=100, output_size=1, l1_lambda=1e-6, activation='relu', seed=42)

epsilon_values = [1e-4, 1e-3, 1e-2]
eta_values = [0.0, 0.1, 0.5, 0.8, 0.9, 0.99]   # 0 => GD classico, ~1 => NAG accelerato
max_iters = 5000
tol = 1e-8

results = []

for eps in epsilon_values:
    for eta in eta_values:
        model = deepcopy(model_original) # copia pulita del modello originale per ogni iter

        history = model.train(X_syn, y_syn, NAG, epsilon = eps, eta = eta, tol = tol, max_iters = max_iters)

        losses = np.array(history["losses"])
        
        f_star = losses[-1] # f* = loss minima (presa dall'ultima iter)
        rel_gap = (losses - f_star) / np.abs(f_star + 1e-15)  # per evitare zero division

        results.append({
            "epsilon": eps,
            "eta": eta,
            "losses": losses,
            "rel_gap": rel_gap
        })

num_eps = len(epsilon_values)
fig, axes = plt.subplots(1, num_eps, figsize=(5 * num_eps, 6), sharey=True)

for i, eps in enumerate(epsilon_values):
    ax = axes[i]

    results_eps = [res for res in results if res["epsilon"] == eps]

    for res in results_eps:
        eta = res["eta"]
        rg = res["rel_gap"]
        iters = np.arange(1, len(rg) + 1)

        label = f"eta={eta}"
        ax.loglog(iters, rg, label=label)

    ref0 = results_eps[0]["rel_gap"][0]  # gap iniziale della prima run con eps corrente
    iters_ref = np.linspace(1, max_iters, 200)
    ref_1_over_t2 = ref0 / (iters_ref**2)
    ref_1_over_t = ref0 / iters_ref

    ax.loglog(iters_ref, ref_1_over_t2, '--', color='brown', label='O(1/t^2)')
    ax.loglog(iters_ref, ref_1_over_t, ':', color='magenta', label='O(1/t)')

    ax.set_xlabel("Iteration (log scale)")
    if i == 0:
        ax.set_ylabel("Relative Gap (log scale)")
    ax.set_title(f"epsilon={eps}")
    ax.legend()
    ax.grid(True)

plt.suptitle("NAG as optimizer for ELM vs. O(1/t^2) e O(1/t)\n momentum (eta) values for different learning rates (epsilon)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("nag_convergence.png", dpi=300)

# ===== Test con lambda diversi =====
from utils import gridSearch_nag_lambda

l1_values = [1e-5, 1e-4, 1e-3, 1e-2]
epsilon_values = [1e-4, 1e-3, 1e-2]
eta_values = [0.0, 0.1, 0.5, 0.8, 0.9, 0.99]

best_params, search_results = gridSearch_nag_lambda(X_syn, y_syn, l1_values, epsilon_values, eta_values)

plt.figure()

for l1_lambda in l1_values:
    lambda_results = [res for res in search_results if res['l1_lambda'] == l1_lambda]
    best_run = min(lambda_results, key=lambda x: x['score'])

    losses = best_run['history']
    final_loss = losses[-1]
    relative_gap = [(loss - final_loss) / final_loss for loss in losses]

    # semilogy = asse y in scala log, asse x lineare
    plt.semilogy(range(1, len(relative_gap) + 1), relative_gap, label=f"lambda={l1_lambda}")

plt.xlabel("Iteration")
plt.ylabel("Relative Gap")
plt.title("Convergence for different $\lambda$")
plt.legend()
plt.savefig("nag_convergence_lambda.png", dpi=300)

# ===== Grid Search finale (per selezionare best epsilon e eta) =====
from utils import gridSearch_nag_best

epsilon_values = [1e-4, 1e-3, 1e-2]
eta_values = [0.0, 0.1, 0.5, 0.8, 0.9, 0.99]
max_iters = 5000
tols = [1e-4, 1e-6, 1e-8, 1e-10]

best_params, best_history = gridSearch_nag_best(X_syn, y_syn, model_original, epsilon_values, eta_values, tols, max_iters)

# ===== California dataset =====
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

l1_values = [1e-5, 1e-4, 1e-3, 1e-2]
epsilon_values = [1e-4, 1e-3, 1e-2]
eta_values = [0.0, 0.1, 0.5, 0.8, 0.9, 0.99]

best_params, search_results = gridSearch_nag_lambda(X_cal, y_cal, l1_values, epsilon_values, eta_values)

plt.figure()
for l1_lambda in l1_values:
    lambda_results = [res for res in search_results if res['l1_lambda'] == l1_lambda]
    best_run = min(lambda_results, key=lambda x: x['score'])
    losses = best_run['history']
    final_loss = losses[-1]

    relative_gap = [(loss - final_loss) / final_loss for loss in losses]
    plt.semilogy(range(1, len(relative_gap) + 1), relative_gap, label=f"lambda={l1_lambda}")

plt.xlabel("Iteration")
plt.ylabel("Relative Gap")
plt.title("Convergence for different $\lambda$ (california)")
plt.legend()
plt.savefig("nag_convergence_lambda_california.png", dpi=300)