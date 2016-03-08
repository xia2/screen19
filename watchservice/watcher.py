import os
import time

class Watcher():
  def __init__(self, files=None, callback=None, **kwargs):
    self._files = files
    self._callback = callback
    self._started = False
    self._running = False

  def is_active(self):
    return self._running

  def start(self, asynchronous=False):
    if self._started:
      return
    self._started = True
    self._running = True
    try:
      while True:
        nextfile = next(self._files)
        while not os.path.exists(nextfile):
          time.sleep(0.1)
        if self._callback:
          self._callback(file=nextfile)
    except StopIteration:
      self._running = False
