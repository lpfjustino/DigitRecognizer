import math
import numpy as np

class Neuron:
    def __init__(self, layer, index, n_weights):
        self.layer = layer
        self.index = index
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=[1, n_weights])[0]
        self.theta = np.random.uniform(low=-0.5, high=0.5)

    def __str__(self):
        parsed_neuron = "\nNeuron " + str(self.index) + " " + self.layer + " layer" + ":\n"
        i = 0

        for w in self.weights:
            parsed_neuron += "w_" + self.layer + "_" + str(self.index) + str(i) + " = \t" + str(w) + "\n"
            i += 1

        parsed_neuron += "theta_" + self.layer + "_" + str(self.index) + " =\t" + str(self.theta) + "\n"

        return parsed_neuron

    def __repr__(self):
        return str(self)

    def as_vector(self):
        return np.append(self.weights, self.theta)


class Layer:
    def __init__(self, layer, n_neurons, prev_layer_neurons):
        self.neurons = [Neuron(layer, i, prev_layer_neurons) for i in range(n_neurons)]

    def __iter__(self):
        return iter(self.neurons)

    def __str__(self):
        parsed_neuron = ""

        for neuron in self.neurons:
            parsed_neuron += str(neuron)

        return parsed_neuron

    def as_matrix(self):
        return np.matrix([neuron.as_vector() for neuron in self.neurons])


class Forward:
    def __init__(self, f_h_net_h_pj, df_h_dnet_h_pj, f_o_net_o_pk, df_o_dnet_o_pk):
        self.f_h_net_h_pj = f_h_net_h_pj
        self.df_h_dnet_h_pj = df_h_dnet_h_pj
        self.f_o_net_o_pk = f_o_net_o_pk
        self.df_o_dnet_o_pk = df_o_dnet_o_pk


class Model:
    def __init__(self, i_neurons=2, h_neurons=20, o_neurons=1):
        self.i_neurons = i_neurons
        self.h_neurons = h_neurons
        self.o_neurons = o_neurons

        self.h_layer = Layer("h", h_neurons, i_neurons)
        self.o_layer = Layer("o", o_neurons, h_neurons)

    def f(self, net):
        return (1 / (1 + (math.exp(-net))))

    def df(self, net):
        return (self.f(net) * (1 - self.f(net)))

    def forward(self, x_p):
        # Compute net and f(net) for every neuron on the hidden layer
        f_h_net_h_pj = np.zeros(self.h_neurons)
        df_h_dnet_h_pj = np.zeros(self.h_neurons)
        for i, neuron in enumerate(self.h_layer):
            net_h_pj = np.dot(np.append(x_p, 1), neuron.as_vector())
            f_h_net_h_pj[i] = self.f(net_h_pj)
            df_h_dnet_h_pj[i] = self.df(net_h_pj)

        # Compute net and f(net) for every neuron on the output layer
        f_o_net_o_pk = np.zeros(self.o_neurons)
        df_o_dnet_o_pk = np.zeros(self.o_neurons)
        for i, neuron in enumerate(self.o_layer):
            net_o_pk = np.dot(np.append(f_h_net_h_pj, 1), neuron.as_vector())
            f_o_net_o_pk[i] = self.f(net_o_pk)
            df_o_dnet_o_pk[i] = self.df(net_o_pk)

        return Forward(f_h_net_h_pj, df_h_dnet_h_pj, f_o_net_o_pk, df_o_dnet_o_pk)

    def classify(self, x_p):
        return self.forward(x_p).f_o_net_o_pk

    def backpropagation(self, X, Y, eta=0.1, threshold=1e-2):
        sqerror = 2 * threshold

        while sqerror > threshold:
            sqerror = 0

            # Train for each given pattern in X knowing its class Y
            for i, x in enumerate(X):
                x_p = X[i]
                y_p = Y[i]

                fwd = self.forward(x_p)
                o_p = fwd.f_o_net_o_pk
                delta_p = y_p - o_p

                sqerror += np.sum(delta_p) ** 2

                # Calculation of output layer's delta for a pattern
                delta_o_p = delta_p * fwd.df_o_dnet_o_pk

                # Calculation of hidden layer's delta for a pattern
                delta_h_p = np.zeros(self.h_neurons)

                # Computes delta for each neuron on the hidden layer
                for j in range(self.h_neurons):
                    sum = 0
                    for k in range(self.o_neurons):
                        sum += delta_o_p[k] * self.o_layer.as_matrix()[k, j]
                    delta_h_p[j] = fwd.df_h_dnet_h_pj[j] * sum

                # Updates the weights for each neuron on the output layer
                for k, neuron in enumerate(self.o_layer):
                    weights = neuron.as_vector()

                    weights += eta * delta_o_p[k] * np.append(fwd.f_h_net_h_pj, 1)

                    neuron.weights = weights[:-1]
                    neuron.theta = weights[len(weights) - 1]


                # Updates the weights for each neuron on the hidden layer
                for j, neuron in enumerate(self.h_layer):
                    weights = neuron.as_vector()

                    weights += eta * delta_h_p[j] * np.append(x_p, 1)

                    neuron.weights = weights[:-1]
                    neuron.theta = weights[len(weights) - 1]

            sqerror /= len(X)
            print(sqerror)
