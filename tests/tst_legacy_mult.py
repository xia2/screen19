#!/usr/bin/env python
# Legacy test in libtbx format

from __future__ import division

def tst_all():
  a = 1
  b = 1
  if a + b == 2:
    print 'OK'
  else:
    raise Exception('maths failed')

def tst_fail():
  a = 1
  b = 1
  if a + b == 3:
    print 'OK'
  else:
    raise Exception('maths works')

if __name__ == '__main__':
  import sys
  if sys.argv[1] == '1':
    tst_all()
  if sys.argv[1] == '2':
    tst_fail()
