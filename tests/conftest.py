from __future__ import absolute_import, division, print_function

import pytest

try:
    from dials_data import *
except ImportError:

    @pytest.fixture
    def dials_data():
        pytest.skip("Test requires python package dials_data")
