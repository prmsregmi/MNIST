import numpy as np
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        self.input = None

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output

    def backward(self, grad_output):
        self.weights_grad = np.dot(self.input.T, grad_output)
        self.bias_grad = np.sum(grad_output, axis=0, keepdims=True)
        # Compute gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def zero_grad(self):
        self.weights_grad.fill(0)
        self.bias_grad.fill(0)

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

class CrossEntropyLoss:
    '''
    Cross-entropy loss with Softmax Calculation
    '''
    def __init__(self):
        self.logits = None
        self.labels = None
        self.softmax = None

    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels
        batch_size = logits.shape[0]
        # Compute shifted logits for numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        # Compute softmax probabilities
        exps = np.exp(shifted_logits)
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)

        # Compute cross-entropy loss
        log_probs = -np.log(self.softmax[range(len(labels)), labels])
        loss = np.sum(log_probs) / batch_size

        return loss

    def backward(self):
        grad_input = self.softmax.copy()
        grad_input[range(len(self.labels)), self.labels] -= 1
        grad_input /= len(self.labels)
        return grad_input
