import random
import time
import stomp
import string
import sys

class MyListener(stomp.ConnectionListener):
  def on_error(self, headers, message):
    print('received an error "%s"' % message)
  def on_message(self, headers, message):
    print('received a message "%s"' % message)

conn = stomp.Connection()
conn.set_listener('', MyListener())
conn.start()
conn.connect('admin', 'password', wait=True)

block = 0
destination = '/queue/test'
key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

print "Producer %s producing into %s" % (key, destination)
while True:
  block += 1
  print "Sending block %s-%d to %s" % (key, block, destination),
  conn.send(body='Block %s-%d' % (key, block), destination=destination)
  print "done"
  time.sleep(1.2)

#conn.subscribe(destination='/queue/test', id=1, ack='auto')
#
#conn.send(body=' '.join(sys.argv[1:]), destination='/queue/test')
#
#time.sleep(2)
#conn.disconnect()
