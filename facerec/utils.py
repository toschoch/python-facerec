#!/usr/bin/python
# -*- coding: UTF-8 -*-
# created: 23.04.2018
# author:  TOS

import logging

log = logging.getLogger(__name__)

from threading import Timer

class RepeatingTimer(object):
    def __init__(self,interval, function, args, kwargs={}):
        super(RepeatingTimer, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.interval = interval
        self._thread = None

    def start(self):
        self.callback()

    def stop(self):
        self.interval = False

    def callback(self):
        if self.interval:
            self.function(*self.args, **self.kwargs)
            self._thread = Timer(self.interval, self.callback)
            self._thread.daemon = True
            self._thread.start()