from ELM import ELM
from optimizers import NAG, smoothed, compute_diameter, compute_mse_lipschitz_constant
from datasets import generate_synthetic_regression_data
import numpy as np
import matplotlib.pyplot as plt
import time

X_syn, y_syn = generate_synthetic_regression_data(n_samples=1000, n_features=50, noise=0.1, random_seed=42)

elm_nag = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
              l1_lambda=0.001, activation='relu', seed=1710)

elm_smoothed = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
                   l1_lambda=0.001, activation='relu', seed=1710)

epochs = 1000

best_nag_params = {
    "epsilon": 0.01, 
    "eta": 0.99,
    "tol": 0.0001,
    "max_iters": epochs
}

print("---[NAG]---")
start_time = time.time()
history_nag = elm_nag.train(X_syn, y_syn, optimizer=NAG, **best_nag_params)
nag_time = time.time() - start_time

elm_smoothed.forward(X_syn)
H = elm_smoothed.hidden_layer_output
D1 = compute_diameter(X_syn)
hidden_repr = elm_smoothed.activation(X_syn @ elm_smoothed.weights_input_hidden + elm_smoothed.bias_hidden)
D2 = compute_diameter(hidden_repr)
M = compute_mse_lipschitz_constant(elm_smoothed, X_syn)

best_smoothed_params = {
    "mu": None,
    "momentum_init": 0.9,
    "D1":D1,
    "D2":D2,
    "M": M,
    "epochs": epochs,
    "nu":None
}
print("---[smoothed]---")
start_time = time.time()
history_smoothed = elm_smoothed.train(X_syn, y_syn,
                                      optimizer=smoothed,
                                      **best_smoothed_params)
smoothed_time = time.time() - start_time

f_star = min(min(history_nag["losses"]), min(history_smoothed["losses"]))

gap_nag = [abs((loss - f_star))/f_star for loss in history_nag["losses"]]
gap_smoothed = [abs((loss - f_star))/f_star for loss in history_smoothed["losses"]]

final_loss_nag = history_nag["losses"][-1]
final_loss_smoothed = history_smoothed["losses"][-1]

print("\n----------------------------")
print("     FINAL RESULTS")
print("----------------------------")
print(f"Final Loss (NAG):       {final_loss_nag:.6f}")
print(f"Final Loss (Smoothed):  {final_loss_smoothed:.6f}")
print(f"Execution Time (NAG):   {nag_time:.4f} seconds")
print(f"Execution Time (Smoothed): {smoothed_time:.4f} seconds")

plt.figure(figsize=(10, 5))
plt.plot(gap_nag, label="NAG", linestyle="-")
plt.plot(gap_smoothed, label="Smoothed", linestyle="--")

plt.xlabel("Iterations")
plt.ylabel("Relative Gap (log scale)")
plt.yscale("log")
plt.title("NAG vs Smoothed: Theoretical $\mu$")
plt.legend()
plt.grid(True)
plt.savefig("nag_vs_smoothed_theoreticalMu.png", dpi=300)

# ------------------------------------------------------------------

epochs = 20000

elm_nag = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
              l1_lambda=0.001, activation='relu', seed=1710)

elm_smoothed = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
                   l1_lambda=0.001, activation='relu', seed=1710)

best_nag_params = {
    "epsilon": 0.01, 
    "eta": 0.99,
    "tol": 0.0001,
    "max_iters": epochs
}

print("---[NAG_2]---")
start_time = time.time()
history_nag = elm_nag.train(X_syn, y_syn, optimizer=NAG, **best_nag_params)
nag_time = time.time() - start_time

elm_smoothed.forward(X_syn)
H = elm_smoothed.hidden_layer_output
D1 = compute_diameter(X_syn)
hidden_repr = elm_smoothed.activation(X_syn @ elm_smoothed.weights_input_hidden + elm_smoothed.bias_hidden)
D2 = compute_diameter(hidden_repr)
M = compute_mse_lipschitz_constant(elm_smoothed, X_syn)

best_smoothed_params = {
    "mu": 1e-3,
    "momentum_init": 0.9,
    "D1":D1,
    "D2":D2,
    "M": M,
    "epochs": epochs,
    "nu":None
}

print("---[smoothed_2]---")
start_time = time.time()
history_smoothed = elm_smoothed.train(X_syn, y_syn,
                                      optimizer=smoothed,
                                      **best_smoothed_params)
smoothed_time = time.time() - start_time

f_star = min(min(history_nag["losses"]), min(history_smoothed["losses"]))

gap_nag = [abs((loss - f_star))/f_star for loss in history_nag["losses"]]
gap_smoothed = [abs((loss - f_star))/f_star for loss in history_smoothed["losses"]]

final_loss_nag = history_nag["losses"][-1]
final_loss_smoothed = history_smoothed["losses"][-1]

print("\n----------------------------")
print("     FINAL RESULTS")
print("----------------------------")
print(f"Final Loss (NAG):       {final_loss_nag:.6f}")
print(f"Final Loss (Smoothed):  {final_loss_smoothed:.6f}")
print(f"Execution Time (NAG):   {nag_time:.4f} seconds")
print(f"Execution Time (Smoothed): {smoothed_time:.4f} seconds")

plt.figure(figsize=(10, 5))
plt.plot(gap_nag, label="NAG", linestyle="-")
plt.plot(gap_smoothed, label="Smoothed", linestyle="--")

plt.xlabel("Iterations")
plt.ylabel("Relative Gap (log scale)")
plt.yscale("log")
plt.title("NAG vs Smoothed: Fixed $\mu=1e-3$")
plt.legend()
plt.grid(True)
plt.savefig("nag_vs_smoothed_fixedMu.png", dpi=300)

# --------------------------------------------------------
# CALIFORNIA
epochs = 1000

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def load_california_housing_data(random_state=42, sample_fraction=0.05):
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    np.random.seed(random_state)
    sample_size = int(sample_fraction * X.shape[0])
    indices = np.random.choice(X.shape[0], sample_size, replace=False)

    return X[indices], y[indices]

X_cal, y_cal = load_california_housing_data()
print(f"X_cal shape: {X_cal.shape}, y_cal shape: {y_cal.shape}")

elm_nag_cal = ELM(input_size=X_cal.shape[1], hidden_size=200, output_size=1,
              l1_lambda=0.001, activation='relu', seed=1710)

elm_smoothed_cal = ELM(input_size=X_cal.shape[1], hidden_size=200, output_size=1,
                   l1_lambda=0.001, activation='relu', seed=1710)

best_nag_params = {
    "epsilon": 0.01,  
    "eta": 0.99,     
    "tol": 0.0001,
    "max_iters": epochs
}

print("---[NAG_cal]---")
start_time = time.time()
history_nag = elm_nag_cal.train(X_cal, y_cal, optimizer=NAG, **best_nag_params)
nag_time = time.time() - start_time


elm_smoothed_cal.forward(X_cal)
H = elm_smoothed_cal.hidden_layer_output
D1 = compute_diameter(X_cal)
hidden_repr = elm_smoothed_cal.activation(X_cal @ elm_smoothed_cal.weights_input_hidden + elm_smoothed_cal.bias_hidden)
D2 = compute_diameter(hidden_repr)
M = compute_mse_lipschitz_constant(elm_smoothed_cal, X_cal)

best_smoothed_params = {
    "mu": 1e-3,
    "momentum_init": 0.9,
    "D1":D1,
    "D2":D2,
    "M": M,
    "epochs": epochs,
    "nu":None
}

print("---[smoothed_cal]---")
start_time = time.time()
history_smoothed = elm_smoothed_cal.train(X_cal, y_cal,
                                      optimizer=smoothed,
                                      **best_smoothed_params)
smoothed_time = time.time() - start_time

f_star = min(min(history_nag["losses"]), min(history_smoothed["losses"]))

gap_nag = [abs((loss - f_star))/f_star for loss in history_nag["losses"]]
gap_smoothed = [abs((loss - f_star))/f_star for loss in history_smoothed["losses"]]

final_loss_nag = history_nag["losses"][-1]
final_loss_smoothed = history_smoothed["losses"][-1]

print("\n----------------------------")
print("     FINAL RESULTS")
print("----------------------------")
print(f"Final Loss (NAG):       {final_loss_nag:.6f}")
print(f"Final Loss (Smoothed):  {final_loss_smoothed:.6f}")
print(f"Execution Time (NAG):   {nag_time:.4f} seconds")
print(f"Execution Time (Smoothed): {smoothed_time:.4f} seconds")

plt.figure(figsize=(10, 5))
plt.plot(gap_nag, label="NAG", linestyle="-")
plt.plot(gap_smoothed, label="Smoothed", linestyle="--")

plt.xlabel("Iterations")
plt.ylabel("Relative Gap (log scale)")
plt.yscale("log")
plt.title("NAG vs fixed mu Smoothed (California)")
plt.legend()
plt.grid(True)
plt.savefig("nag_vs_smoothed_california.png", dpi=300)

# --------------------------------------------------------
from scipy.optimize import minimize
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

epochs = 20000

elm_nag = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
              l1_lambda=0.001, activation='relu', seed=1710)

elm_smoothed = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
                   l1_lambda=0.001, activation='relu', seed=1710)

best_nag_params = {
    "epsilon": 0.01,  
    "eta": 0.99,  
    "tol": 0.0001,
    "max_iters": epochs
}

print("---[NAG_final]---")
start_time = time.time()
history_nag = elm_nag.train(X_syn, y_syn, optimizer=NAG, **best_nag_params)
nag_time = time.time() - start_time



elm_smoothed.forward(X_syn)
H = elm_smoothed.hidden_layer_output
D1 = compute_diameter(X_syn)
hidden_repr = elm_smoothed.activation(X_syn @ elm_smoothed.weights_input_hidden + elm_smoothed.bias_hidden)
D2 = compute_diameter(hidden_repr)
M = compute_mse_lipschitz_constant(elm_smoothed, X_syn)

best_smoothed_params = {
    "mu": 1e-3,
    "momentum_init": 0.9,
    "D1":D1,
    "D2":D2,
    "M": M,
    "epochs": epochs,
    "nu":None}

print("---[smoothed_final]---")
start_time = time.time()
history_smoothed = elm_smoothed.train(X_syn, y_syn,
                                      optimizer=smoothed,
                                      **best_smoothed_params)
smoothed_time = time.time() - start_time

#################### ADAM #########################
def train_with_adam(model, X, y,
                        lr=1e-3,
                        epochs=100,
                        batch_size=None,
                        verbose=True,
                        activation='relu',
                        tol=1e-6):

        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

        W_in_const = tf.constant(model.weights_input_hidden, dtype=tf.float32)
        b_in_const = tf.constant(model.bias_hidden, dtype=tf.float32)

        W_out_var = tf.Variable(model.weights_hidden_output, dtype=tf.float32)
        b_out_var = tf.Variable(model.bias_output, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        if isinstance(activation, str):
            if activation == 'relu':
                activation_tf = tf.nn.relu
            elif activation == 'tanh':
                activation_tf = tf.nn.tanh
            else:
                raise ValueError("Unsupported activation string for TF.")
        else:
            raise ValueError("Custom Python activation not supported in train_with_adam demo.")

        def forward_tf(X_):
            hidden = activation_tf(tf.matmul(X_, W_in_const) + b_in_const)
            output = tf.matmul(hidden, W_out_var) + b_out_var
            return output

        N = X.shape[0]
        if batch_size is None or batch_size >= N:
            batch_size = N

        loss_history = []
        num_batches = (N + batch_size - 1) // batch_size
        best_loss = float("inf")


        for epoch in range(epochs):
            idx = tf.random.shuffle(tf.range(N))
            epoch_loss = 0.0

            for i in range(num_batches):
                batch_idx = idx[i*batch_size:(i+1)*batch_size]
                X_batch = tf.gather(X_tf, batch_idx)
                y_batch = tf.gather(y_tf, batch_idx)

                with tf.GradientTape() as tape:
                    preds = forward_tf(X_batch)
                    mse = 0.5 * tf.reduce_mean((preds - y_batch)**2)
                    l1_term = model.l1_lambda * tf.reduce_sum(tf.abs(W_out_var))
                    loss = mse + l1_term

                grads = tape.gradient(loss, [W_out_var, b_out_var])
                optimizer.apply_gradients(zip(grads, [W_out_var, b_out_var]))
                epoch_loss += loss.numpy()

            epoch_loss /= num_batches
            loss_history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

            # Early stopping check
            if epoch_loss < best_loss - tol:
                best_loss = epoch_loss

        model.weights_hidden_output = W_out_var.numpy()
        model.bias_output = b_out_var.numpy()

        return loss_history

elm_adam = ELM(input_size=X_syn.shape[1], hidden_size=200, output_size=1,
               l1_lambda=0.001, activation='relu', seed=1710)

print("---[ADAM]---")
start_time = time.time()
history_adam = train_with_adam(elm_adam, X_syn, y_syn, lr=0.01, epochs=epochs, activation='relu', verbose=False)
adam_time = time.time() - start_time

f_star = min(min(history_nag["losses"]), min(history_smoothed["losses"]), min(history_adam))
gap_nag = [abs((loss - f_star)) / f_star for loss in history_nag["losses"]]
gap_smoothed = [abs((loss - f_star)) / f_star for loss in history_smoothed["losses"]]
gap_adam = [abs((loss - f_star)) / f_star for loss in history_adam]

print("\n----------------------------")
print("     FINAL RESULTS")
print("----------------------------")
print(f"Final Loss (NAG):       {history_nag['losses'][-1]:.6f}")
print(f"Final Loss (Smoothed):  {history_smoothed['losses'][-1]:.6f}")
print(f"Final Loss (Adam):      {history_adam[-1]:.6f}")
print(f"Execution Time (NAG):   {nag_time:.4f} seconds")
print(f"Execution Time (Smoothed): {smoothed_time:.4f} seconds")
print(f"Execution Time (Adam):  {adam_time:.4f} seconds")

plt.figure(figsize=(10, 5))
plt.plot(gap_nag, label="NAG", linestyle="-")
plt.plot(gap_smoothed, label="Smoothed", linestyle="--")
plt.plot(gap_adam, label="Adam (Keras)", linestyle=":")

plt.xlabel("Iterations")
plt.ylabel("Relative Gap (log scale)")
plt.yscale("log")
plt.title("NAG vs Smoothed vs Adam (Keras)")
plt.legend()
plt.grid(True)
plt.savefig("nag_vs_smoothed_vs_adam.png", dpi=300)