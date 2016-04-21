from __future__ import division
import libtbx.load_env
from libtbx import test_utils
from libtbx.test_utils.pytest import discover

tst_list = [
#  "$D/tests/tst_legacy.py",
  ["$D/tests/tst_legacy_mult.py", "1"]
#  ["$D/tests/tst_legacy_mult.py", "2"]
] + discover("i19")

if (__name__ == "__main__"):
  build_dir = libtbx.env.under_build("i19")
  dist_dir = libtbx.env.dist_path("i19")
  test_utils.run_tests(build_dir, dist_dir, tst_list)
