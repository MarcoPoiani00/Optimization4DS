import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from itertools import product
from ELM import ELM
from optimizers import NAG, smoothed, compute_diameter, compute_mse_lipschitz_constant
import json
from copy import deepcopy

def gridSearch_nag_lambda(X, y, l1_values, epsilon_values, eta_values):
    best_score = float('inf')
    best_params = None
    results = []

    param_combinations = list(product(l1_values, epsilon_values, eta_values))

    with tqdm(total=len(param_combinations), desc="progress") as pbar:
        for l1_lambda, epsilon, eta in param_combinations:
            model = ELM(
                input_size=X.shape[1],
                hidden_size=200,
                output_size=1,
                l1_lambda=l1_lambda,
                activation='relu',
                seed=42
            )

            start_time = time.time()
            history = model.train(
                X, y,
                optimizer=NAG,
                epsilon=epsilon,
                eta=eta,
                tol=1e-6,
                max_iters=1000
            )
            end_time = time.time()

            final_loss = history['losses'][-1]  # loss finale
            num_iterations = len(history['losses'])
            time_taken = end_time - start_time

            score = final_loss + time_taken

            results.append({
                'l1_lambda': l1_lambda,
                'epsilon': epsilon,
                'eta': eta,
                'final_loss': final_loss,
                'time_taken': time_taken,
                'num_iterations': num_iterations,
                'score': score,
                'history': history['losses']
            })

            if score < best_score:
                best_score = score
                best_params = (l1_lambda, epsilon, eta)

            pbar.update(1)

    return best_params, results

def gridSearch_nag_best(X_syn, y_syn, model_original, epsilon_values, eta_values, tols, max_iters):
    best_score = float("inf")
    best_params = None
    best_history = None

    for eps in epsilon_values:
        for eta in eta_values:
            for tol in tols:
                model = deepcopy(model_original)

                start_time = time.time()
                history = model.train(X_syn, y_syn, NAG, epsilon=eps, eta=eta, tol=tol, max_iters=max_iters)
                end_time = time.time()

                final_loss = history["losses"][-1]
                elapsed = end_time - start_time

                score = final_loss + elapsed

                if score < best_score:
                    best_score = score
                    best_params = {
                        "epsilon": eps,
                        "eta": eta,
                        "tol": tol,
                        "max_iters": max_iters,
                        "elapsed": elapsed,
                        "final_loss": final_loss,
                        "score": score
                    }

                    best_history = history
    
    print("\n=== Risultati Grid Search su NAG ===")
    print(f"Miglior configurazione trovata: {best_params}")
    print(f"Loss finale: {best_params['final_loss']}")
    print(f"Tempo di esecuzione: {best_params['elapsed']} sec")
    print(f"Score combinato: {best_params['score']}")

    with open("best_nag_config.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nConfigurazione migliore di NAG salvata in 'best_nag_config.json'.")

    return best_params, best_history

def grid_search_smoothed_lambda(X, y, lambda_values=None, epochs=1000, momentum=0.9):

    if lambda_values is None:
        lambda_values = {
            'lambda 1e-5': 1e-5,
            'lambda 1e-4': 1e-4,
            'lambda 1e-3': 1e-3,
            'lambda 1e-2': 1e-2
        }

    results = {}

    for lambda_key, lambda_val in lambda_values.items():
        print(f"Running experiment for: {lambda_key}")

        model = ELM(input_size=X.shape[1], hidden_size=200, output_size=1,
                    l1_lambda=lambda_val, activation='relu', seed=1710)

        model.forward(X)
        H = model.hidden_layer_output
        D1 = compute_diameter(X)
        hidden_repr = model.activation(X @ model.weights_input_hidden + model.bias_hidden)
        D2 = compute_diameter(hidden_repr)
        M = compute_mse_lipschitz_constant(model, X)

        start_time = time.time()
        history = model.train(
            X, y,
            D1=D1,
            D2=D2,
            optimizer=smoothed,
            epochs=epochs,
            M=M,
            momentum_init=momentum,
            gradient_norm_threshold=1e-6,
            verbose = False
        )

        execution_time = time.time() - start_time

        results[lambda_key] = {
            'final_loss': history["losses"][-1],
            'loss_history': history["losses"],
            'grad_norms': history["grad_norms"],
            'execution_time': execution_time
        }

        print(f"Final Loss: {history['losses'][-1]:.6f}, Execution Time: {execution_time:.2f} sec\n")

    best_lambda = min(results, key=lambda k: results[k]['final_loss'])
    best_loss = results[best_lambda]['final_loss']

    return (best_lambda, best_loss), results

def grid_search_hyperparams_smoothed(X, y, param_grid, epochs=1000, optimizer=smoothed):

    param_combinations = list(itertools.product(*param_grid.values()))
    results = {}

    for params in tqdm(param_combinations, desc="Grid Search Progress", leave=True):
        momentum_init, mu, nu = params
        param_key = f"momentum={momentum_init}, mu={mu}, nu={nu}"

        print(f"Running experiment for: {param_key}")

        model = ELM(input_size=X.shape[1], hidden_size=200, output_size=1,
                    l1_lambda=0.001, activation='relu', seed=1710)

        model.forward(X)
        H = model.hidden_layer_output
        D1 = compute_diameter(X)
        hidden_repr = model.activation(X @ model.weights_input_hidden + model.bias_hidden)
        D2 = compute_diameter(hidden_repr)
        sigma_1 = 1.0
        sigma_2 = np.min(np.linalg.eigvals(H.T @ H))
        M = compute_mse_lipschitz_constant(model, X)

        start_time = time.time()
        history = model.train(
            X, y,
            D1=D1,
            D2=D2,
            optimizer=optimizer,
            epochs=epochs,
            M=M,
            momentum_init=momentum_init,
            mu=mu,
            nu=nu,
            gradient_norm_threshold=1e-6,
            verbose = False
        )
        execution_time = time.time() - start_time

        results[param_key] = {
            'final_loss': history["losses"][-1],
            'loss_history': history["losses"],
            'grad_norms': history["grad_norms"],
            'execution_time': execution_time
        }

        print(f"Final Loss: {history['losses'][-1]:.6f}, Execution Time: {execution_time:.2f} sec")

    best_params = min(results, key=lambda k: results[k]['final_loss'])
    best_loss = results[best_params]['final_loss']

    with open("best_smoothed_config.json", "w") as f:
        json.dump(results[best_params], f, indent=2)
    print("\nConfigurazione migliore di smoothed salvata in 'best_smoothed_config.json'.")

    return best_params, best_loss, results