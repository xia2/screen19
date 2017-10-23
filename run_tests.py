from __future__ import absolute_import, division, print_function
from libtbx.test_utils.pytest import discover

tst_list = [
#  "$D/tests/tst_legacy.py",
  ["$D/tests/tst_legacy_mult.py", "1"]
#  ["$D/tests/tst_legacy_mult.py", "2"]
] + discover()
