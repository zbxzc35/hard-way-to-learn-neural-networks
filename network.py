# -*- coding:utf-8 -*- 
from __future__ import division
__author__ = 'Dragonfly'
import random
import numpy as np
"""
In this section, we compute the sample one by one
"""


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # y is a vector, we create it by np.random.randn(y) instead of np.random.randn(y, 1)
        self.bias = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # a is single example
        for w, b in zip(self.weights, self.bias):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0} : {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.bias = [b - (eta/len(mini_batch*nb)) for b, nb in zip(self.bias, nabla_b)]
        self.weights = [w - (eta/len(mini_batch)*nw) for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        delta_nabla_b = [np.zeros(b.shape) for b in self.bias]
        delta_nabla_w = [np.zeros(b.shape) for w in self.weights]
        activation = x
        activatioins = [activation]
        zs = []
        # forward
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, x) + b
            zs.append(z)
            activation = sigmoid(z)
            activatioins.append(activation)
        # back
        delta = self.cost_derivative(activatioins[-1], y) * sigmoid_prime(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activatioins[-2].T)
        for l in xrange(2, self.num_layers):
            delta = np.dot(self.weights[-l].T, delta) * sigmoid_prime(zs[-l])
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activatioins[-l-1].T)
        return delta_nabla_b, delta_nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == '__main__':
    pass