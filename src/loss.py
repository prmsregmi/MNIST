import numpy as np

class CrossEntropyLoss:
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
        log_probs = -np.log(self.softmax[np.arange(batch_size), labels] + 1e-15)
        loss = np.sum(log_probs) / batch_size
        return loss

    def backward(self):
        batch_size = self.logits.shape[0]
        grad_input = self.softmax.copy()
        grad_input[np.arange(batch_size), self.labels] -= 1
        grad_input /= batch_size
        return grad_input
