#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-19 08:49:51
import Connection
import Connections
import Layer


class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = []
        '''
        初始化各个layer，layer初始化的时候，就初始化了node
        '''
        self.connections = Connections()
        for i in range(len(layers)):
            self.layers.append(Layer(i, layers[i]))
        '''
        除最后一层之外，各层之间加入connection
        '''
        for layer in len(layers - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in
                           self.layers[layer + 1].nodes[:-1]]
            '''
            构建connection的上下游节点
            '''
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, label, data_set, rate, iteration):
        for i in range(iteration):
            for d in range(len(label)):
                self.train_one_sample(label[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        计算预测值向量
        '''
        self.predict(sample)
        '''
        计算增量
        '''
        self.calc_delta(label)
        '''
        更新权重
        '''
        self.update_weight(rate)

    def predict(self, sample):
        '''
        初始化第一层参数，也就是输入向量x
        '''
        self.layers[0].set_output(sample)
        '''
        后面各个层，分别计算output
        '''
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        '''
        返回最后一层的output作为计算结果
        '''
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        '''
        计算输出层每一个node的增量
        '''
        for i in range(len(label)):
            output_nodes.calc_output_layer_delta(label[i])
        '''
        计算隐藏曾节点的增量
        '''
        for layer in self.layers[-2:: -1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        更新所有的connection的weight
        '''
        for layer in self.layers[: -1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()
