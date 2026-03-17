from Layer import Layer
import pickle
import os
import random
from dataset import *

save_path="model.plk"

class Network:
    def __init__(self, layer_size=[32, VOCAB_SIZE]):
        self.layers= []
        for size in layer_size:
            self.layers.append(Layer(size))

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, expected_value, learning_rate):
        # Backprob
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                layer.backward_output(expected_value)
            else:
                layer.backward_hidden(self.layers[::-1][i - 1])
        # update weights
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def train(self, dataset, save=None, epochs=500, learning_rate=1):
        samples = [sample for line in dataset for sample in line]
        if not samples:
            return

        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss = 0.0

            for sample in samples:
                x = sample[0]
                y_true = sample[1]
                y_pred = self.forward(x)
                # print(f"{x} - {y_true} - {y_pred}")

                if len(y_pred) != len(y_true):
                    raise ValueError("Output size does not match target size")

                loss = sum([0.5*((p - t)**2) for p, t in zip(y_pred, y_true)])
                total_loss += loss
                self.backward(y_true, learning_rate)

            average_loss = total_loss / len(samples)
            print(f"EPOCH {epoch} - total_loss = {average_loss}")
        if(save):
            with open("model.plk", "wb") as f:
                pickle.dump(self, f, 0)


if __name__ == "__main__":
    random.seed(3)

    dataset = read_dataset(path=os.path.join(os.path.dirname(__file__), "input_small.txt"))

    train_dataset = dataset[:150]

    test_dataset = dataset[:50]
    if os.path.isfile(save_path):
        with open("model.plk", "rb") as f:
            network = pickle.load(f)
    else:
        network = Network()

    network.train(train_dataset, save="model.plk", epochs=6)
    print(detokenize([network.forward(tokenize("W"))]))
    print(detokenize([network.forward(tokenize("We"))]))
    print(detokenize([network.forward(tokenize("We "))]))
    print(detokenize([network.forward(tokenize("We a"))]))
    print(detokenize([network.forward(tokenize("We ar"))]))

    # predictions = []
    # for inp in test_input:
    #     predictions.append(network.forward(inp))
    # print(test_output)
    # print(predictions)
    # print(detokenize(test_output))
    # print(detokenize(predictions))
    # print(detokenize([network.forward(tokenize("j")[0])]))