from __future__ import division
import libtbx.load_env

def discover_pytests(module):
  try:
    import os
    import pytest
  except ImportError:
    def pytest_warning():
      print "=" * 60
      print "WARNING: Skipping some tests\n"
      print "To run all available tests you need to install pytest"
      print "eg. with   libtbx.python -m pip install pytest"
      print "=" * 60
    pytest_warning()
    import atexit
    atexit.register(pytest_warning)
    return []
  if 'LIBTBX_SKIP_PYTEST' in os.environ:
    return []

  test_list = []
  dist_dir = libtbx.env.dist_path(module)
  class TestDiscoveryPlugin:
    def pytest_itemcollected(self, item):
      test_list.append([ "libtbx.python", "-m", "pytest", '--noconftest',
        os.path.join(dist_dir, item.nodeid) ])
  print "Discovering pytest tests:"
  pytest.main(['-qq', '--collect-only', '--noconftest', dist_dir], plugins=[TestDiscoveryPlugin()])
  return test_list

if (__name__ == "__main__"):
  import unittest
  test_suite = unittest.defaultTestLoader.discover(libtbx.env.dist_path("i19"), pattern="tst_*.py")
  result = unittest.TextTestRunner().run(test_suite)
  import sys
  sys.exit(0 if result.wasSuccessful() else 1)

tst_list = [
#  "$D/tests/tst_legacy.py",
  ["$D/tests/tst_legacy_mult.py", "1"]
#  ["$D/tests/tst_legacy_mult.py", "2"]
] + discover_pytests("i19")
