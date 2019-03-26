from __future__ import absolute_import, division, print_function

import pytest
from screen19.screen import Screen19


def test_screen19_command_line_help_does_not_crash():
    Screen19().run([])


def test_screen19(dials_data, tmpdir):
    data_dir = dials_data("x4wide").strpath
    Screen19().run([data_dir])

    logfile = run_in_tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile
    assert "photon incidence rate is outside the linear response region" in logfile


@pytest.mark.xfail(raises=ValueError, reason="LAPACK bug?")
def test_screen19_single_frame(dials_data, run_in_tmpdir):
    # TODO Use a single frame with fewer than 80 reflections
    data_dir = dials_data("x4wide")
    image = data_dir.join("X4_wide_M1S4_2_0001.cbf").strpath

    Screen19().run([image])

    logfile = run_in_tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile
