import numpy as np

class ELM:
    def __init__(self, input_size, hidden_size, output_size,
                 l1_lambda=0.0, activation='relu', seed=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1_lambda = l1_lambda

        if seed is not None:
            np.random.seed(seed)

        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = self._relu
                self.activation_deriv = self._relu_derivative
            elif activation.lower() == 'tanh':
                self.activation = self._tanh
                self.activation_deriv = self._tanh_derivative
            else:
                raise ValueError("Unsupported activation string. Use 'relu' or 'tanh'.")
        else:
            self.activation = activation
            self.activation_deriv = None

        # Randomly init input->hidden (fixed in ELM)
        if isinstance(activation, str) and activation.lower() == 'relu':
            # He initialization for ReLU
            he_std = np.sqrt(2.0 / input_size)
            self.weights_input_hidden = np.random.normal(
                loc=0.0, scale=he_std, size=(input_size, hidden_size)
            )
            self.bias_hidden = np.random.normal(
                loc=0.0, scale=1e-2, size=(1, hidden_size)
            )
        else:
            # Tanh => Xavier initialization
            limit = 0.2 * np.sqrt(6.0 / (input_size + hidden_size))
            self.weights_input_hidden = np.random.uniform(
                low=-limit, high=limit, size=(input_size, hidden_size)
            )
            self.bias_hidden = np.random.uniform(
                low=-limit, high=limit, size=(1, hidden_size)
            )

        # Hidden->output weights (trainable) + bias
        limit_out = 0.2 * np.sqrt(6.0 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.uniform(
            low=-limit_out, high=limit_out, size=(hidden_size, output_size)
        )
        self.bias_output = np.random.uniform(
            low=-limit_out, high=limit_out, size=(1, output_size)
        )

        self.hidden_layer_input = None
        self.hidden_layer_output = None
        self.output_layer_input = None
        self.predicted_output = None

    # ACTIVATION
    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        y = np.tanh(x)
        return 1.0 - y*y

    # L1 subgradient
    def _l1_subgrad(self, w):
        grad = np.zeros_like(w)
        grad[w > 0] = 1.0
        grad[w < 0] = -1.0
        return grad

    # Forward/Backward pass
    def forward(self, X):
        self.hidden_layer_input = X.dot(self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.activation(self.hidden_layer_input)
        self.output_layer_input = self.hidden_layer_output.dot(self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.output_layer_input  # linear output
        return self.predicted_output

    def backward(self, X, y):
        n_samples = X.shape[0]
        output_error = (self.predicted_output - y)
        dW2 = (self.hidden_layer_output.T @ output_error) / n_samples
        db2 = np.sum(output_error, axis=0, keepdims=True) / n_samples

        # add L1 subgradient
        dW2 += self.l1_lambda * self._l1_subgrad(self.weights_hidden_output)
        return dW2, db2

    def evaluate_loss(self, X, y):
        pred = self.forward(X)
        mse = 0.5 * np.mean((pred - y)**2)
        l1_term = self.l1_lambda * np.sum(np.abs(self.weights_hidden_output))
        return mse + l1_term

    def predict(self, X):
        hidden = self.activation(X.dot(self.weights_input_hidden) + self.bias_hidden)
        return hidden.dot(self.weights_hidden_output) + self.bias_output

    # parameters utils
    def _pack_params(self, W_out, b_out): #flatten and concatenate trainable parameters into one vector
        return np.concatenate([W_out.ravel(), b_out.ravel()])

    def _unpack_params(self, theta): # unpack flattened vector theta into ELM's W_out, b_out shapes.
        w_size = self.hidden_size * self.output_size
        w = theta[:w_size].reshape((self.hidden_size, self.output_size))
        b = theta[w_size:].reshape((1, self.output_size))

        self.weights_hidden_output = w
        self.bias_output = b

    # TRAINING
    def train(self, X, y, optimizer, **optimizer_args):
        """
        Train ELM using a custom optimizer with signature:
            optimizer( f, theta_init, ... ) -> (theta_opt, history)

        'f(theta)' returns (loss, grad, None).
        """
        def objective(theta):
            self._unpack_params(theta)
            self.forward(X)

            loss = 0.5 * np.mean((self.predicted_output - y) ** 2)
            loss += self.l1_lambda * np.sum(np.abs(self.weights_hidden_output))

            dW2, db2 = self.backward(X, y)

            grad = self._pack_params(dW2, db2)
            return loss, grad, None

        theta_init = self._pack_params(self.weights_hidden_output, self.bias_output)
        theta_opt, history = optimizer(objective, theta_init, **optimizer_args)
        self._unpack_params(theta_opt)

        return history

    def get_parameters(self):
        return {
            "weights_hidden_output": self.weights_hidden_output,
            "bias_output": self.bias_output
        }

    def set_parameters(self, params):
        self.weights_hidden_output = params["weights_hidden_output"]
        self.bias_output = params["bias_output"]