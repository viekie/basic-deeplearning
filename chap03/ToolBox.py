#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-19 14:13:49


def gradient_checker(network, sample_feature, sample_label):

    def network_error(vec1, vec2):
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    network.get_gradient(sample_feature, sample_label)

    for conn in network.connections.connections:
        actual_gradient = conn.get_gradient()

        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        conn.weight -= 2 * epsilon
        error2 = network_error(network.predict(sample_feature), sample_label)

        expect_gradient = (error2 - error1) / 2 * epsilon

        print 'actual_gradient: %f\r\n expect_gradient: %f \r\n' % \
            (actual_gradient, expect_gradient)
