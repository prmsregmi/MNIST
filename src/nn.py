import numpy as np
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        self.input = None

    def forward(self, x):
        self.input = x
        output = np.dot(x, self.weights) + self.bias
        return output

    def backward(self, grad_output):
        self.weights_grad = np.dot(self.input.T, grad_output)
        self.bias_grad = np.sum(grad_output, axis=0, keepdims=True)
        # Compute gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def zero_grad(self):
        self.weights_grad.fill(0)
        self.bias_grad.fill(0)
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
            grad = np.clip(grad, -1, 1)
            v = self.velocities[id(param)]
            v_new = self.momentum * v - self.learning_rate * grad
            self.velocities[id(param)] = v_new
            param += v_new