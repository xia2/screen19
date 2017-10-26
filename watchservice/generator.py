from __future__ import absolute_import, division, print_function

import itertools

def Generator(pattern, begin, end):
  '''A generator for monotonically increasing filenames following a predifined
     pattern.'''
  return itertools.imap(
      lambda i: pattern % i,
      xrange(begin, end + 1)
    )
