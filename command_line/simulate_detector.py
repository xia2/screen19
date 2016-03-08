# creates empty .cbf files at set speed

import itertools
import time

pattern = 'somefile_01_%05d.cbf'
speed = 0.01 # seconds / image
r = xrange(1800)

start = time.time()

for image, timewait in itertools.izip(r, itertools.count(start, speed)):
  while time.time() < timewait:
    time.sleep(speed / 10)
  filename = pattern % image
  open(filename, 'a').close()
  print filename
