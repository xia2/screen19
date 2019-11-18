from __future__ import absolute_import, division, print_function

import pytest
from screen19.screen import Screen19


def test_screen19_command_line_help_does_not_crash():
    Screen19().run([])


def test_screen19(dials_data, tmpdir):
    data_dir = dials_data("x4wide").strpath
    with tmpdir.as_cwd():
        Screen19().run([data_dir], set_up_logging=True)

    logfile = tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile
    assert "photon incidence rate is outside the linear response region" in logfile


data_files = [
    [""],
    ["x3_1_00*.cbf.gz"],
    ["x3_1_####.cbf.gz:1:99"],
    ["x3_1_00##.cbf.gz:1:99"],
    ["x3_1_0001.cbf.gz:1:99"],
    [
        "x3_1_0001.cbf.gz",
        "x3_1_0002.cbf.gz",
        "x3_1_0003.cbf.gz",
        "x3_1_0004.cbf.gz",
        "x3_1_0005.cbf.gz",
    ],
]


@pytest.mark.parametrize("data_files", data_files)
def test_screen19_inputs(dials_data, tmpdir, data_files):
    """Test various valid input argument styles"""
    data = [
        dials_data("small_molecule_example").join(filename).strpath
        for filename in data_files
    ]

    with tmpdir.as_cwd():
        Screen19().run(data, set_up_logging=True)

    logfile = tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile


@pytest.mark.xfail(raises=ValueError, reason="LAPACK bug?")
def test_screen19_single_frame(dials_data, tmpdir):
    # TODO Use a single frame with fewer than 80 reflections
    image = dials_data("x4wide").join("X4_wide_M1S4_2_0001.cbf").strpath

    with tmpdir.as_cwd():
        Screen19().run([image], set_up_logging=True)

    logfile = tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile
