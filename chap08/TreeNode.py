#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-09 14:19:10


class TreeNode(object):

    def __init__(self, data, children=[], children_data=[]):
        self.parent = None
        self.children = children
        self.childen_data = children_data
        self.data = data
        for child in children:
            child.parent = self
