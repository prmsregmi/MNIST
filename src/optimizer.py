from .model import LinearLayer
import numpy as np

class SGDWithMomentum:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        # Initialize velocities
        for layer in self.model:
            if isinstance(layer, LinearLayer):
                self.velocities[id(layer.weights)] = np.zeros_like(layer.weights)
                self.velocities[id(layer.bias)] = np.zeros_like(layer.bias)

    def step(self):
        for layer in self.model:
            if isinstance(layer, LinearLayer):
                v = self.velocities[id(layer.weights)]
                self.velocities[id(layer.weights)] = (self.momentum * v - self.learning_rate * layer.weights_grad)
                layer.weights += self.velocities[id(layer.weights)]
                v = self.velocities[id(layer.bias)]
                self.velocities[id(layer.bias)] = (self.momentum * v - self.learning_rate * layer.bias_grad)
                layer.bias += self.velocities[id(layer.bias)]

    def zero_grad(self):
        for layer in self.model:
            if isinstance(layer, LinearLayer):
                layer.weights_grad.fill(0)
                layer.bias_grad.fill(0)