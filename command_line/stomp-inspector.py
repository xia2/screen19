import time
import stomp
import sys

conn = stomp.Connection()

subscriptionid=1

class MyListener(stomp.ConnectionListener):
  def on_error(self, headers, message):
    print('received an error "%s"' % message)
  def on_message(self, headers, message):
    print('seen a message "%s"' % message)
#    time.sleep(5)

conn.set_listener('', MyListener())
conn.start()
conn.connect('admin', 'password', wait=True)

conn.subscribe(destination='/queue/test', id=subscriptionid, ack='auto', browser='true')

time.sleep(10)

conn.disconnect()
