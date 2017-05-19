#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-17 13:21:19
import math


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0.0
        self.delta = 0.0

    def set_output(self, output):
        self.output = output

    def append_upstream_connections(self, conn):
        '''
        插入上游指向当前node的connection
        '''
        self.upstream.append(conn)

    def append_downstream_connections(self, conn):
        '''
        插入下游节点node的connection
        '''
        self.downstream.append(conn)

    def calc_output(self):
        '''
        计算当前节点输出，也就是sum(conni.weight * conn.upstream.node.output)
        最用在进行sigmoid计算
        '''
        output = reduce(lambda ret, conn:
                        ret + conn.weight * conn.upstream_node.output,
                        self.upstream,
                        0.0)
        self.output = self.sigmoid(output)

    def sigmoid(self, value):
        return 1.0 / (1.0 + math.exp(0 - value))

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(lambda ret, conn: ret
                                  + conn.downstream_node.delta * conn.weight,
                                  self.downstream,
                                  0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        打印node本身
        '''
        layer_info = 'layer_index: %d' % self.layer_index
        node_info = 'node_index: %s' % self.node_index
        downstream_info = 'downstream: %s' % self.downstream
        upstream_info = 'upstream: %s' % self.upstream
        output_info = 'output: %s' % self.output

        return layer_info + '\r\n' + node_info + '\r\n' + downstream_info + \
            '\r\n' + upstream_info + '\r\n' + output_info
