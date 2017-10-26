from __future__ import absolute_import, division, print_function

import itertools
import os
import shutil
import sys
import time

# creates (copies) .cbf files at set speed

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

srcpattern = '/dls/tmp/wra62962/jeppe-analysis/data/04/nidppecl2_d_phiscans_04_%05d.cbf'
dstpattern = 'image_%05d.cbf'
speed = 0.1 # seconds / image
r = xrange(1, 3600)

start = time.time()

for image, timewait in itertools.izip(r, itertools.count(start, speed)):
  in_time = True
  while time.time() < timewait:
    time.sleep(speed / 10)
    in_time = False
  shutil.copyfile(srcpattern % image, dstpattern % image)
  sys.stdout.write('+' if in_time else '-')
