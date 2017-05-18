#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-18 09:51:21


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        self.delta = 0.0

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        rest = reduce(lambda ret, conn: ret +
                      conn.dowstream_node.delta * conn.weight,
                      self.downstream, 0.0)

        self.delta = rest * self.output * (1 - self.output)

    def __str__(self):
        layer_info = 'layer_index: %d\r\n' % self.layer_index
        node_info = 'node_index: %d\r\n' % self.node_index
        downstream_info = 'down_stream: %s\r\n' % self.downstream
        delta_info = 'delta: %f\r\n' % self.delta
        output_info = 'output: %d\r\n' % self.output

        return layer_info + node_info \
            + downstream_info + delta_info + output_info
