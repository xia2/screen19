from __future__ import absolute_import, division, print_function

from dials.conftest import run_in_tmpdir

import pytest

try:
    from dials_data import *
except ImportError:

    @pytest.fixture
    def dials_data():
        pytest.skip("Test requires python package dials_data")
