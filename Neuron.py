import random
from maths import *

class Neuron:
    def __init__(self, biais=None):
        self.weights = None

        if biais == None:
            self.biais = random.uniform(-1, 1)
        else:
            self.biais = biais
        
        self.last_input = []
        self.output = 0
        self.delta = 0

    def forward(self, items):
        # Keep the last input for weight updates
        self.last_input = items

        # Ensure weights match the input size.
        # If new input dimensions appear, extend weights with small random values.
        if self.weights is None:
            self.weights = [random.uniform(-1, 1) for _ in range(len(items))]
        elif len(self.weights) < len(items):
            extra = [random.uniform(-1, 1) for _ in range(len(items) - len(self.weights))]
            self.weights.extend(extra)

        # Compute output using the available weights (ignore extra weights if input is smaller).
        self.output = sum([item * self.weights[i] for i, item in enumerate(items)]) + self.biais
        self.output = sigmoid(self.output)
        return self.output
    
    def calculate_delta(self, error):
        self.delta = error * derived_sigmoid(self.output)
    
    def update_weights(self, learning_rate):
        # Only update weights that correspond to the last input size.
        # This avoids index errors when the input length changes during training.
        for i in range(min(len(self.weights), len(self.last_input))):
            self.weights[i] += learning_rate * self.delta * self.last_input[i]
        
        self.biais += self.delta * learning_rate
       
