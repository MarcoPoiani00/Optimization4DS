import numpy as np

def NAG(f, theta, epsilon, eta, tol, max_iters, grad_norm_threshold=None):
    """
    params:
        f (function): Function that returns (value, gradient, Hessian).
        theta (np.array): Initial parameter vector.
        epsilon (float): Learning rate.
        eta (float): Momentum coefficient.
        tol (float): Convergence tolerance.
        max_iters (int): Maximum number of iterations.
        patience (int): Number of iterations without improvement before stopping.
        loss_threshold (float): Upper bound to stop if loss explodes.
        grad_norm_threshold (float, optional): Stopping criterion based on gradient norm.

    returns:
        theta (np.array): Optimized parameters.
        history (dict): Contains losses, gradient norms, and trajectory.
    """
    v = np.zeros_like(theta)
    history = {"losses": [], "grad_norms": [], "trajectory": []}

    for t in range(1, max_iters + 1):
        theta_lookahead = theta + eta * v
        loss, grad, _ = f(theta_lookahead)
        if grad is None or np.isnan(loss):
            print(f"Arresto: gradiente None o loss NaN a iterazione {t}.")
            break

        grad_norm = np.linalg.norm(grad)
        history["losses"].append(loss)
        history["grad_norms"].append(grad_norm)

        if grad_norm < tol or (grad_norm_threshold and grad_norm < grad_norm_threshold):
            print(f"Convergenza a iterazione {t}, norma gradiente = {grad_norm:.6f}")
            break

        v = eta * v - epsilon * grad
        theta += v
        history["trajectory"].append(theta.copy())

    return theta, history

def optimizer_wrapper_nag(f):
    f_name = f.__name__

    if f_name == "ackley":
        theta_init = np.array([0.2, 0.2])
    elif f_name == "rosenbrock":
        theta_init = np.array([-1.2, 1.0])
    else:
        theta_init = np.array([-2.0, 2.0])

    eta = 0.9
    epsilon = 0.001
    tol = 1e-5
    max_iters = 10000

    _, history = NAG(
        f, 
        theta=theta_init, 
        epsilon=epsilon, 
        eta=eta, 
        tol=tol, 
        max_iters=max_iters
    )
    return history

def smoothed(
    f,
    theta_init,
    A_norm = 1.0, # identity matrix
    D1 = 1.0,
    D2 = 1.0,
    sigma_1 = 1.0,
    sigma_2 = 1.0,
    epochs=1000,
    M=None,
    mu=None,
    nu=None,
    momentum_init=0.9,
    gradient_norm_threshold=None,
    verbose=True
):
    assert A_norm > 0, "A_norm must be positive."
    assert D1 > 0 and D2 > 0, "D1 and D2 must be positive."
    assert sigma_1 > 0 and sigma_2 > 0, "sigma_1 and sigma_2 must be positive."

    if mu is None:
        mu = (2 * A_norm / (epochs + 1)) * np.sqrt(D1 / (sigma_1 * sigma_2 * D2))
    assert mu > 0, "Computed mu must be positive."

    L_L1 = (1.0 / (mu * sigma_2)) * A_norm
    if M is None:
        M = L_L1

    L_mu = M + L_L1
    if nu is None:
        nu = 1.0 / L_mu

    if verbose:
        print("------ PARAMETERS ------")
        print(f"M = {M}")
        print(f"D1 = {D1}, D2 = {D2}")
        print(f"sigma_1 = {sigma_1}, sigma_2 = {sigma_2}")
        print(f"mu = {mu}")
        print(f"L_L1 = {L_L1}")
        print(f"L_mu = {L_mu}")
        print(f"nu (learning rate) = {nu}")
        print("------------------------")

    v = np.zeros_like(theta_init)
    theta = theta_init.copy()
    eta = momentum_init

    history = {"losses": [], "grad_norms": [], "trajectory": [theta.copy()]}

    for epoch in range(epochs):
        theta_look = theta + eta * v
        loss, grad, _ = f(theta_look)

        if grad is None or np.isnan(loss):
            print(f"Arresto: gradiente None o loss NaN a epoca {epoch}.")
            break

        gn = np.linalg.norm(grad)
        history["losses"].append(loss)
        history["grad_norms"].append(gn)

        if gradient_norm_threshold is not None and gn < gradient_norm_threshold:
            print(f"Convergenza a epoca {epoch}, norma gradiente < {gradient_norm_threshold}")
            break
        
        v = eta * v - nu * grad
        theta = theta + v
        history["trajectory"].append(theta.copy())

    return theta, history

def compute_mse_lipschitz_constant(model, X):
    hidden_inp = X.dot(model.weights_input_hidden) + model.bias_hidden
    H = model.activation(hidden_inp)
    sigma_max = np.linalg.norm(H, ord=2)  # spectral norm

    M = 2*(sigma_max**2)/X.shape[0]

    return M

def compute_diameter(Q):
    return np.max(np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1))