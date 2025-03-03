import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

class ELM:
    """
    Extreme Learning Machine for a single-hidden-layer feedforward neural network (SLFN).
    Steps (Huang et al., 2006):
      1) Randomly assign input weights W1 and biases b for the hidden layer.
      2) Compute hidden layer output matrix H.
      3) Compute output weights beta = H^dagger * T (Moore-Penrose pseudoinverse).
    """
    def __init__(self, input_size, hidden_size, output_size, l1_lambda=0.0, activation='relu', seed=None):
        """
        Parameters
        ----------
        input_size : int
            Number of input features (dimension of x).
        hidden_size : int
            Number of hidden neurons.
        output_size : int
            Number of output dimensions (dimension of t).
        activation : callable, optional
            Activation function g(z) to use. 
            If None, defaults to ReLU.
        seed : int, optional
            Seed for reproducible random initialization.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1_lambda = l1_lambda  # Add this line
        
        if seed:
            np.random.seed(seed)

        # Choose activation
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
            # user-supplied function
            self.activation = activation
            # no derivative provided => handle carefully or raise error
            self.activation_deriv = None
            print("Warning: no derivative for a custom activation. Backprop may fail.")
        
        # Randomly init input->hidden weights, not updated in ELM
        # For ReLU, a good approach is He initialization:
        # For Tanh, a good approach is Xavier (scaled uniform).
        if isinstance(activation, str) and activation.lower() == 'relu':
            # He initialization
            he_std = np.sqrt(2.0 / input_size)
            self.weights_input_hidden = np.random.normal(
                loc=0.0, scale=he_std, size=(input_size, hidden_size)
            )
            self.bias_hidden = np.random.normal(
                loc=0.0, scale=1e-2, size=(1, hidden_size)
            )
        else:
            # e.g. Tanh => Xavier
            limit = np.sqrt(6.0 / (input_size + hidden_size))
            self.weights_input_hidden = np.random.uniform(
                low=-limit, high=limit, size=(input_size, hidden_size)
            )
            self.bias_hidden = np.random.uniform(
                low=-limit, high=limit, size=(1, hidden_size)
            )
        
        # Hidden->output weights: We DO train these
        # We'll do a simple Xavier-like approach for either ReLU or Tanh:
        limit_out = np.sqrt(6.0 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.uniform(
            low=-limit_out, high=limit_out, size=(hidden_size, output_size)
        )
        self.bias_output = np.random.uniform(
            low=-limit_out, high=limit_out, size=(1, output_size)
        )

        # Placeholders for forward pass
        self.hidden_layer_input = None
        self.hidden_layer_output = None
        self.output_layer_input = None
        self.predicted_output = None

    # ------------------------
    # Activation functions
    # ------------------------
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        # derivative wrt the pre-activation
        return (x > 0).astype(float)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        # derivative wrt pre-activation for tanh
        # if we define y = tanh(x), derivative = 1 - y^2
        y = np.tanh(x)
        return 1.0 - y*y
    
    # L1 subgradient
    def _l1_subgrad(self, w):
        # returns sign(w), with sign(0)=0
        grad = np.zeros_like(w)
        grad[w > 0] = 1.0
        grad[w < 0] = -1.0
        return grad
    
    # ------------------------
    # Forward pass
    # ------------------------
    def forward(self, X):
        """
        Forward pass with either ReLU or tanh hidden activation,
        then a linear activation (or if you prefer, you could 
        also do tanh at the output).
        """
        # hidden
        self.hidden_layer_input = X.dot(self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.activation(self.hidden_layer_input)
        
        # output
        self.output_layer_input = self.hidden_layer_output.dot(self.weights_hidden_output) + self.bias_output
        # We'll do linear output by default. 
        # If you want tanh final, do: self.predicted_output = np.tanh(self.output_layer_input)
        self.predicted_output = self.output_layer_input
        
        return self.predicted_output

    # ------------------------
    # Backward pass
    # ------------------------
    def backward(self, X, y):
        """
        Compute gradients wrt the hidden->output weights 
        for a MSE + L1 penalty on W2.
        
        Because we do not update input->hidden in an ELM 
        (by definition it's random and fixed), 
        we only compute partial derivatives wrt (W2, b2).
        """
        n_samples = X.shape[0]
        
        # 1) dLoss/d(output)
        # MSE derivative: (pred - y)
        output_error = (self.predicted_output - y)  # shape (n_samples, output_size)
        
        # 2) derivative wrt W2, b2
        # hidden_layer_output shape: (n_samples, hidden_size)
        dW2 = (self.hidden_layer_output.T @ output_error) / n_samples
        db2 = np.sum(output_error, axis=0, keepdims=True) / n_samples
        
        # L1 subgradient on W2
        if self.l1_lambda > 1e-15:
            dW2 += self.l1_lambda * self._l1_subgrad(self.weights_hidden_output)
        
        return dW2, db2

    # ------------------------
    # Update weights
    # ------------------------
    def update(self, dW2, db2, lr=1e-3):
        """
        Gradient descent step on hidden->output layer
        """
        self.weights_hidden_output -= lr * dW2
        self.bias_output -= lr * db2

    # ------------------------
    # Evaluate
    # ------------------------
    def evaluate_loss(self, X, y):
        """
        Return MSE + L1 penalty for the forward pass.
        MSE = 0.5 * mean( (pred - y)^2 )
        plus L1 = lambda * sum(|W2|)
        ignoring W1 since not trained.
        """
        pred = self.forward(X)
        mse = 0.5 * np.mean((pred - y)**2)
        l1_term = self.l1_lambda * np.sum(np.abs(self.weights_hidden_output))
        return mse + l1_term
    
    def predict(self, X):
        """
        Just forward pass, ignoring state variables
        """
        hidden = self.activation(X.dot(self.weights_input_hidden) + self.bias_hidden)
        # linear output
        output = hidden.dot(self.weights_hidden_output) + self.bias_output
        return output
