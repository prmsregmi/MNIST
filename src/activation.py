import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        output = np.maximum(0, x)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        # Numerically stable with large exponentials
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # Assuming that grad_output is the gradient of the loss w.r.t. softmax output
        batch_size = grad_output.shape[0]
        grad_input = np.empty_like(grad_output)
        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            grad_input[i] = np.dot(jacobian, grad_output[i])
        return grad_input