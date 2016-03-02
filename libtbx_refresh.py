from __future__ import division

try:
  import i19.util.version as version
  print version.i19_version()
except ImportError:
  pass
