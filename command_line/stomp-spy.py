import json
import time
import stomp
import sys

class GDAspy(stomp.ConnectionListener):
  def __init__(self):
    self._conn = stomp.Connection([('sci-serv5.diamond.ac.uk', 61613)])
    self._queues = {
      'status': '/queue/scisoft.xia2.STATUS_QUEUE',
#     'status-topic': '/topic/scisoft.xia2.STATUS_TOPIC',
      'submission': '/queue/scisoft.xia2.SUBMISSION_QUEUE'
    }
    self._conn.set_listener('', self)
    self._conn.start()
    self._conn.connect(wait=True)

    self._queues = {
      q: {'queue': self._queues[q],
          'id': n,
          'list': [],
          'closed': False}
        for q, n in zip(self._queues.iterkeys(), range(len(self._queues)))
    }

    for q in self._queues.itervalues():
      print "Subscribing to %s (%d)" % (q['queue'], q['id'])
      self._conn.subscribe(destination=q['queue'], id=q['id'], ack='auto', browser='true')

    while any( not q['closed'] for q in self._queues.itervalues()):
      time.sleep(0.2)
#   self._conn.disconnect()
    print "All information read.\n\n"

    print "xia2 status queue:"
    for (header, body) in self._queues['status']['list']:
      if 'browser' not in header:
        result = self.parse_as_json(body)
        if result is None:
          continue
        print " %s: %s" % (result['status'], result['name'])
        for k in sorted(result.iterkeys()):
          if k != 'status' and k != 'name':
            print "  %s: %s" % (k, result[k])
      print

    print "xia2 submission queue:"
    for (header, body) in self._queues['submission']['list']:
      if 'browser' not in header:
        result = self.parse_as_json(body)
        if result is None:
          continue
        for k in sorted(result.iterkeys()):
          print "  %s: %s" % (k, result[k])
      print

  def on_error(self, headers, message):
    print "Received an error"
    print headers
    print message
    sys.exit(1)

  def on_message(self, headers, message):
#   print "Received message on queue %s" % headers['subscription']
    for q in self._queues.itervalues():
      if q['id'] == int(headers['subscription']):
        q['list'].append((headers, message))
        q['closed'] = headers.get('browser', '') == 'end'
        break

  def parse_as_json(self, message):
    try:
      return json.loads(message)
    except Exception:
      return None

GDAspy()


