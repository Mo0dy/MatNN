import numpy as np


# some way to store weights and biases in layers (matrices)
# activation function
# some way to do forward propagation (calculation)


class NN(object):
    def __init__(self, input_layer_size, hidden_layer_size, hidden_layers_amount, output_layer_size):
        self.layers = []
        # the last row is the biases
        self.layers.append(self.random_layer(input_layer_size + 1, hidden_layer_size))
        for i in range(1, hidden_layers_amount):
            self.layers.append(self.random_layer(hidden_layer_size + 1, hidden_layer_size))
        self.layers.append(self.random_layer(hidden_layer_size + 1, output_layer_size))

    def random_layer(self, m, n):
        return (np.random.rand(m, n) - 0.5) * 2

    def activation_function(self, values):
        return np.tanh(values)

    def calculate(self, inputs):
        # next_layer = activation_func(prev_layer(appended 1) * weights_next_layer)
        a = inputs
        for l in self.layers:
            # a is the result of the calculation
            # append for the biases calculation
            a = self.activation_function(np.append(a, 1) @ l)
        return a

# genetic algorithm
# normalize data (automatically)

if __name__ == "__main__":
    print("neural network test")
    nn = NN(2, 5, 2, 2)
    print(nn.calculate(np.random.rand(2)))
