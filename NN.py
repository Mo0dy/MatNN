import numpy as np


# some way to store weights and biases in layers (matrices)
# activation function
# some way to do forward propagation (calculation)


class NN(object):
    def __init__(self, input_layer_size, hidden_layer_size, hidden_layers_amount, output_layer_size):
        # genetic algorithm export and import preperation
        shapes = [[input_layer_size + 1, hidden_layer_size]]
        for i in range(1, hidden_layers_amount):
            shapes.append([hidden_layer_size + 1, hidden_layer_size])
        shapes.append([hidden_layer_size + 1, output_layer_size])
        self.shapes = np.array(shapes)

        # create layers
        self.layer_amount = hidden_layers_amount + 1
        self.layers = [self.random_layer(shapes[i]) for i in range(self.layer_amount)]

        self.layer_sizes = self.shapes[:, 0] * self.shapes[:, 1]

        # plus biases is used to calculate the genome length
        self.total_weights = np.sum(self.layer_sizes)

    def random_layer(self, shape):
        return (np.random.rand(shape[0], shape[1]) - 0.5) * 2

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

    # for genetic algorithm
    def export_genome(self):
        genome = np.zeros(self.total_weights)
        current_index = 0
        for i in range(self.layer_amount):
            genome[current_index: current_index + self.layer_sizes[i]] = self.layers[i].flatten()
            current_index += self.layer_sizes[i]
        return genome

    def import_genome(self, genome):
        current_index = 0
        for i in range(self.layer_amount):
            self.layers[i] = genome[current_index: current_index + self.layer_sizes[i]].reshape(self.shapes[i])
            current_index += self.layer_sizes[i]


# genetic algorithm
# normalize data (automatically)

if __name__ == "__main__":
    print("neural network test")
    nn = NN(2, 5, 2, 2)
    print(nn.calculate([1, 2]))
    genome = nn.export_genome()
    nn.import_genome(genome)
    genome2 = nn.export_genome()

    print(np.all(genome == genome2))

