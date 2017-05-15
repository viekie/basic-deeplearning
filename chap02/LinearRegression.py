#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-12 15:31:07


class LinearRegression(object):

    def __init__(self, input_num, activitor):
        '''
        construction function
        '''
        self.weights = [0 for _ in range(input_num)]
        self.bais = 0.0
        self.activitor = activitor

    def __str__(self):
        return 'weights: %s, bais:%s' % (self.weights, self.bais)

    def predict(self, sample_attr):
        return reduce(lambda a, b: a + b,
                      map(lambda (x, w): x * w,
                          zip(sample_attr, self.weights)),
                      self.bais)

    def train(self, samples_attr, samples_label, iteration, rate):
        for i in range(iteration):
            self.update_model_weights(samples_attr, samples_label, rate)

    def update_model_weights(self, samples_attr, samples_label, rate):
        for (sample_attr, sample_label) in zip(samples_attr, samples_label):
            output = self.predict(sample_attr)
            self.update(output, sample_attr, sample_label, rate)

    def update(self, output, sample_attr, sample_label, rate):
        delta = sample_label - output
        self.weights = map(lambda(w, x): w + delta * x * rate,
                           zip(self.weights, sample_attr))
        self.bais += rate * delta


def activitor(x):
    return x


def prepare_train_data():
    samples_attr = [[5], [3], [8], [1.4], [10.1]]
    samples_labels = [5500, 2300, 7600, 1800, 11400]
    return samples_attr, samples_labels


def get_linear_regression_model(input_num, iteration, rate):
    lr = LinearRegression(input_num, activitor)
    samples_vector, samples_label = prepare_train_data()
    lr.train(samples_vector, samples_label, iteration, rate)
    return lr


if __name__ == '__main__':
    lr = get_linear_regression_model(1, 10, 0.01)
    print lr
    print 'work 3.4 years, salary is %s', lr.predict([3.4])
    print 'work 15 years, salary is %s', lr.predict([15])
    print 'work 1.5 years, salary is %s', lr.predict([1.5])
    print 'work 6.3 years, salary is %s', lr.predict([6.3])
