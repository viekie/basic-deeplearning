#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-18 20:23:29
import Node
import ConstNode


class Layer(object):
    def __init__(self, layer_index, node_num):
        '''
        每个层有自己的层号， 每个层包含一系列的node， 最后增加一个ConstNode
        '''
        self.layer_index = layer_index
        self.node = []
        for i in range(node_num):
            self.node.append(Node(layer_index, i))

        self.node.append(ConstNode(layer_index, node_num))

    def set_output(self, data):
        for i in len(data):
            self.node[i].set_output(data[i])

    def calc_output(self):
        for node in self.node[:-1]:
            node.calc_output()

    def __str__(self):
        layer_info = 'layer_info: %d\r\n' % self.layer_index
        node_info = 'node_info: %s\r\n' % self.node
        return layer_info + node_info
