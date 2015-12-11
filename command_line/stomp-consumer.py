import time
import stomp
import sys

conn = stomp.Connection()

subscriptionid=1

class MyListener(stomp.ConnectionListener):
  def on_error(self, headers, message):
    print('received an error "%s"' % message)
  def on_message(self, headers, message):
    print('received a message "%s"' % message)
    conn.ack(headers['message-id'], subscriptionid)
    time.sleep(5)


conn.set_listener('', MyListener())
conn.start()
conn.connect('admin', 'password', wait=True)

conn.subscribe(destination='/queue/test', id=subscriptionid, ack='client-individual', headers={'activemq.prefetchSize':1})

while True:
  time.sleep(2)

conn.disconnect()
