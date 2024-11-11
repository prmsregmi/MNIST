import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.labels = None

    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels
        batch_size = logits.shape[0]
        # Convert labels to one-hot encoding
        labels_one_hot = np.zeros_like(logits)
        labels_one_hot[np.arange(batch_size), labels] = 1
        # Compute loss
        loss = -np.sum(labels_one_hot * np.log(logits + 1e-15)) / batch_size
        return loss

    def backward(self):
        batch_size = self.logits.shape[0]
        labels_one_hot = np.zeros_like(self.logits)
        labels_one_hot[np.arange(batch_size), self.labels] = 1
        grad_input = (self.logits - labels_one_hot) / batch_size
        return grad_input