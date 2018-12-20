from __future__ import absolute_import, division, print_function

import os
import random
import time


class Watcher:
    def __init__(self, files=None, callback=None, basespeed=0.1):
        self._files = files
        self._callback = callback
        self._started = False
        self._running = False
        self._basespeed = basespeed
        self._waitlimit = 2048

    def _terminate(self):
        self._running = False

    def is_active(self):
        return self._running

    def start(self, asynchronous=False):
        if self._started:
            return
        self._started = True
        self._running = True
        firstfile = True
        try:
            while True:
                nextfile = next(self._files)
                backoff = self._basespeed
                total_wait = 0
                while not os.path.exists(nextfile):
                    if backoff >= self._waitlimit * 2:
                        if self._callback:
                            self._callback(
                                file=nextfile, success=False, wait=total_wait
                            )
                        self._terminate()
                        return
                    if backoff > self._waitlimit:
                        waittime = self._waitlimit
                    else:
                        waittime = random.uniform(self._basespeed, backoff)
                    if firstfile:
                        waittime = min(3, waittime)
                    time.sleep(waittime)
                    total_wait += waittime
                    if firstfile:
                        backoff += 40
                    else:
                        backoff *= 2

                if self._callback:
                    self._callback(file=nextfile, success=True, wait=total_wait)
            firstfile = False
        except StopIteration:
            self._terminate()
            return
