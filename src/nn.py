import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros((1, output_dim))
        # Initialize gradients
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        # Placeholder for input data
        self.input = None

    def forward(self, x):
        self.input = x  # Save input for backward pass
        output = np.dot(x, self.weights) + self.bias
        return output

    def backward(self, grad_output):
        # Compute gradients
        self.weights_grad = np.dot(self.input.T, grad_output)
        self.bias_grad = np.sum(grad_output, axis=0, keepdims=True)
        # Compute gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

class SGDWithMomentum:
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        # Initialize velocities
        for param in parameters:
            self.velocities[id(param)] = np.zeros_like(param)

    def step(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            v = self.velocities[id(param)]
            v_new = self.momentum * v - self.learning_rate * grad
            self.velocities[id(param)] = v_new
            param += v_new

    def zero_grad(self, gradients):
        for grad in gradients:
            grad.fill(0)
