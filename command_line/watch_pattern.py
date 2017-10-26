from __future__ import division

import json
import sys
from optparse import SUPPRESS_HELP, OptionParser

import stomp
from i19.watchservice.generator import Generator
from i19.watchservice.watcher import Watcher as FileWatcher

class Watch():
  def __init__(self, upstream=None):
    self._upstream = upstream

  def watch(self):
    rng = self._opts.range.split(',')
    g = Generator(self._opts.pattern, int(rng[0]), int(rng[1]))
    w = FileWatcher(files=g, callback=self._report)
    w.start(asynchronous=False)
    pass

  def _report(self, file=None, success=False, wait=0, **kwargs):
    if success and self._opts.verbose:
      print "Image %s arrived after %.2f seconds" % (file, wait)
    if not success:
      print "Image %s failed to arrive after %.2f seconds" % (file, wait)
    if self._upstream:
      self._upstream.image_arrived(file=file, success=success, wait=wait, **kwargs)

  def run(self):
    parser = OptionParser()
    parser.add_option("-p", "--pattern", metavar='PATTERN', type='string',
                      nargs=1, dest="pattern",
                      help="Pattern of files to look out for")
    parser.add_option("-r", "--range", metavar='RANGE', type='string',
                      nargs=1, dest="range",
                      help="Range of file numbers, eg. -r 1,1800")
    parser.add_option("-v", action="store_true", dest="verbose", default=False,
                      help="be moderately verbose")
    parser.add_option("-?", help=SUPPRESS_HELP, action="help")
    self._opts, args = parser.parse_args()

    if self._opts.pattern and self._opts.range:
      self.watch()

#class MyListener(stomp.ConnectionListener):
#  def on_error(self, headers, message):
#    print('received an error "%s"' % message)
#  def on_message(self, headers, message):
#    print('received a message "%s"' % message)

class Stomp():
  def __init__(self):
    self._conn = stomp.Connection([('i19-1-control', 61613)])
#    self._conn.set_listener('', MyListener())
    self._conn.set_listener('', stomp.PrintingListener())
    self._conn.start()
    self._conn.connect('i19-filewatcher', '', wait=True)

  def image_arrived(self, **kwargs):
    self._conn.send(
      body=json.dumps(kwargs),
      destination = '/topic/i19-image-arrived'
    )

if __name__ == "__main__":
  Watch(upstream=Stomp()).run()

#block = 0
#destination = '/queue/test'
#key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
#
#print "Producer %s producing into %s" % (key, destination)
#while True:
#  block += 1
#  print "Sending block %s-%d to %s" % (key, block, destination),
#  conn.send(body='Block %s-%d' % (key, block), destination=destination)
#  print "done"
#  time.sleep(1.2)
#
#conn.subscribe(destination='/queue/test', id=1, ack='auto')
#
#conn.send(body=' '.join(sys.argv[1:]), destination='/queue/test')
#
#time.sleep(2)
#conn.disconnect()
