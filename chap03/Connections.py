#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-19 08:43:35


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def __str__(self):
        conn_info = 'connections: %s' % self.connections
        return conn_info
