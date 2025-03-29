import numpy as np
import matplotlib.pyplot as plt

class TestFunctions:
    def __init__(self):
        self.functions = [
            self.quadratic,
            self.rosenbrock,
            self.ackley
        ]

        self.global_minima = {
            "quadratic": np.array([0, 0]),
            "rosenbrock": np.array([1, 1]),
            "ackley": np.array([0, 0])
        }

    def quadratic(self, x, Q=None, q=None):
        """ Quadratic function: f(x) = 0.5 * x'Qx + q'x """
        if Q is None:
            Q = np.array([[2, 0], [0, 2]])  # default positive definite matrix
        if q is None:
            q = np.array([0, 0])

        if x is None:
            if np.min(np.linalg.eigvals(Q)) > 1e-14:
                x_star = np.linalg.solve(Q, -q)
                v = 0.5 * x_star.T @ Q @ x_star + q.T @ x_star
            else:
                v = -np.inf
            return v, np.array([0, 0]), Q
        else:
            v = 0.5 * x.T @ Q @ x + q.T @ x
            g = Q @ x + q
            H = Q
            return v, g, H

    def rosenbrock(self, x):
        if x is None:
            return 0, np.array([-1, 1]), np.array([[2, 0], [0, 200]])

        v = 100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2
        g = np.array([
            2*x[0] - 400*x[0] * (x[1] - x[0]**2) - 2,
            -200*x[0]**2 + 200*x[1]
        ])
        H = np.array([
            [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
            [-400*x[0], 200]
        ])

        return v, g, H

    def ackley(self, x):
        if x is None:
            return 0, np.array([2, 2]), None

        a, b, c = 20, 0.2, 2 * np.pi
        sum_sq = np.sum(x**2) / len(x)
        sum_cos = np.sum(np.cos(c * x)) / len(x)
        v = -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.exp(1)

        grad = a * b * np.exp(-b * np.sqrt(sum_sq)) * (x / np.sqrt(sum_sq) / len(x)) + (c / len(x)) * np.exp(sum_cos) * np.sin(c * x)

        return v, grad, None

    def plot_contours(self, optimizer):

        colors = ['grey', 'blue', 'green']
        fig, axes = plt.subplots(1, len(self.functions), figsize=(18, 6))
        fig.suptitle("Optimization Paths on Test Functions Level Sets", fontsize=16)

        for i, (f, ax) in enumerate(zip(self.functions, axes)):
            history = optimizer(f)
            trajectory = np.array(history["trajectory"])
            x_min, x_max = np.min(trajectory[:, 0]) - 0.05, np.max(trajectory[:, 0]) + 0.05
            y_min, y_max = np.min(trajectory[:, 1]) - 0.05, np.max(trajectory[:, 1]) + 0.05

            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_vals, y_vals)

            Z = np.zeros_like(X)
            for rr in range(X.shape[0]):
                for cc in range(X.shape[1]):
                    v, _, _ = f(np.array([X[rr, cc], Y[rr, cc]]))
                    Z[rr, cc] = v

            ax.contour(X, Y, Z, levels=30, cmap='viridis')
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                    marker='o', color=colors[i],
                    label=f'{f.__name__.capitalize()} Path')
            
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color='black', marker='s', label='Start')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='yellow', marker='*', label='End')

            global_min = self.global_minima[f.__name__]
            ax.scatter(global_min[0], global_min[1], color='red', marker='x', label='Global Min', s=200)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{f.__name__.capitalize()}", fontsize=14)
            ax.legend()

            last_iter = len(history["losses"])
            last_loss = history["losses"][-1] if history["losses"] else float('nan')
            last_grad_norm = history["grad_norms"][-1] if history["grad_norms"] else float('nan')
            diff = np.linalg.norm(trajectory[-1] - global_min)

            ax.text(0.5, -0.3, f"Last Iter: {last_iter}\nLast Loss: {last_loss:.4f}\nLast Grad Norm: {last_grad_norm:.4f}\nDiff from Global Min: {diff:.4f}",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)

        plt.tight_layout()
        # save figure without overwriting existing
        plt.savefig(f"test_functions_contours_{optimizer.__name__}.png", dpi=300)