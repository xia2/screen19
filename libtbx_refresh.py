from __future__ import division

try:
  import i19.util.version as version
  print version.i19_version()
except ImportError:
  pass

try:
  import libtbx.pkg_utils
  libtbx.pkg_utils.require('mock', '>=2.0')
  libtbx.pkg_utils.require('pytest', '>=2')
except ImportError:
  print "\n" * 10 + "Could not verify dependencies: cctbx sources out of date" + "\n" * 10
