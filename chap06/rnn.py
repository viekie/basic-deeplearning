#!/usr/bin/env python
# -*- coding: utf8 -*-
# Power by viekie2017-10-06 10:17:09

import numpy as np


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


class ReluActivator(object):
    def forward(self, weighted_input):
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class RecurrentLayer(object):
    def __init__(self, input_width, state_width,
                 activator, learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate

        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros([state_width, 1]))

        self.U = np.random.uniform(-1e-4, 1e-4, (state_width, input_width))
        self.W = np.random.uniform(-1e-4, 1e-4, (state_width, state_width))

    def forward(self, input_array):
        self.times += 1
        state = (np.dot(self.U, input_array) +
                 np.dot(self.W, self.state_list[-1]))
        element_wise_op(state, self.activator.forward)

    def backward(self, sentivity_array, activator):
        self.calc_delta(sentivity_array, activator)
        self.calc_gradient()

    def update(self):
        self.W -= self.learning_rate * self.gradient

    def calc_delta(self, sentivity_array, activator):
        self.delta_list = []
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width, 1)))
        self.delta_list.append(sentivity_array)

        for k in range(self.times - 1, 0, -1):
            self.calc_delta_k(k, activator)

    def calc_delta_k(self, k, activator):
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1],
                        activator.backward)
        self.delta_list[k] = np.dot(
            np.dot(self.delta_list[k+1].T, self.W),
            np.diag(state[:, 0])).T

    def calc_gradient(self):
        self.gradient_list = []
        for t in range(self.times+1):
            self.gradient_list.append(np.zeros((self.state_width,
                                                self.state_width)))
            for t in range(self.times, 0, -1):
                self.calc_gradient_t(t)
            self.gradient = reduce(lambda a, b:
                                   a + b,
                                   self.gradient_list,
                                   self.gradient_list[0])

    def calc_gradient_t(self, t):
        gradient = np.dot(self.delta_list[t],
                          self.state_list[t-1].T)
        self.gradient_list[t] = gradient

    def reset_state(self):
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((self.state_width, 1)))


def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d


if __name__ == '__main__':
    layer = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
    x, d = data_set()
    layer.forward(x[0])
    layer.forward(x[1])
    layer.backward(d, ReluActivator())
