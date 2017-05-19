#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-19 16:36:23
import sys
from datatime import datatime
import ImageLoader
import LabelLoader
sys.path.append('../')

from chap03 import NeuralNetwork


def get_train_datasets():
    image_loader = ImageLoader('train-images-idx3-ubyte', 60000)
    label_loader = LabelLoader('train-labels-idx3-ubyte', 60000)
    return image_loader.load(), label_loader.load()


def get_test_datasets():
    image_loader = ImageLoader('t10k-images-idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-labels-idx3-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def get_result(vec):
    '''
    预测输出的向量，最大值作为预测结果
    '''
    max_value = 0.0
    index = 0

    for i in range(len(vec)):
        if (vec[i]) > max_value:
            max_value = vec[i]
            index = i
    return index


def evaluate(network, test_dataset, test_labels):
    error = 0
    total = len(test_dataset)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_dataset[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def train_and_eval():
    last_error_ratio = 1.0
    epoch = 0
    train_datasets, train_labels = get_train_datasets()
    test_datasets, test_labels = get_test_datasets()

    network = NeuralNetwork([784, 350, 10])

    while True:
        epoch += 1
        network.train(train_labels, train_datasets, 0.2, 100)
        print 'time: %s, finish: %d' % (datatime.now(), epoch)
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_datasets, test_labels)
            print 'time: %s, epoch: %d, error ratio: %f' % (datatime.now(),
                                                            epoch,
                                                            error_ratio)
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


if __name__ == '__main__':
    train_and_eval()
