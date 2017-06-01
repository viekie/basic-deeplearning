#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-01 11:00:45


class IdentityActivator(object):
    def forward(self, input):
        return input

    def backward(self, input):
        return 1
