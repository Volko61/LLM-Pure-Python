from Neuron import Neuron

class Layer:
    def __init__(self, number_neurons=10):
        self.neurons = [Neuron(biais=0) for _ in range(number_neurons)]

    def forward(self, input):
        res =  [neuron.forward(input) for neuron in self.neurons]
        return res
    
    def backward_output(self, expected_output):
        for i, neuron in enumerate(self.neurons):
            error = expected_output[i] - neuron.output
            neuron.calculate_delta(error)
    
    def backward_hidden(self, next_layer):
        for i, neuron in enumerate(self.neurons):
            error = 0
            for next_neuron in next_layer.neurons:
                error += next_neuron.weights[i] * next_neuron.delta
            neuron.calculate_delta(error)
    
    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)