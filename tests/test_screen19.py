# coding: utf-8

from __future__ import absolute_import, division, print_function

import pytest

from screen19 import minimum_exposure
from screen19.screen import Screen19

# A list of tuples of example sys.argv[1:] cases and associated image count.
import_checks = [
    ([""], 900),
    (["/"], 900),
    (["x3_1_00*.cbf.gz"], 99),
    (["x3_1_####.cbf.gz:1:99"], 99),
    (["x3_1_00##.cbf.gz:1:99"], 99),
    (["x3_1_0001.cbf.gz:1:99"], 99),
    (
        [
            "x3_1_0001.cbf.gz",
            "x3_1_0002.cbf.gz",
            "x3_1_0003.cbf.gz",
            "x3_1_0004.cbf.gz",
            "x3_1_0005.cbf.gz",
        ],
        5,
    ),
]


def test_screen19_command_line_help_does_not_crash():
    Screen19().run([])


def test_minimum_exposure_help_does_not_crash():
    minimum_exposure.run(args=[])


@pytest.mark.parametrize("import_checks", import_checks)
def test_screen19_inputs(dials_data, tmpdir, import_checks):
    """Test various valid input argument styles"""
    data_files, image_count = import_checks
    data = [
        dials_data("small_molecule_example").join(filename).strpath
        for filename in data_files
    ]

    foo = Screen19()
    # The tmpdir should only be necessary for DIALS v1 — no output expected for DIALS v2
    foo._import(data)

    # Check that the import has resulted in the creation of a single experiment.
    assert len(foo.expts) == 1
    # Check that the associated imageset has the expected number of images.
    assert foo.expts[0].imageset.size() == image_count


def test_screen19(dials_data, tmpdir):
    """An integration test.  Check the full functionality of screen19."""
    data_dir = dials_data("x4wide").join("X4_wide_M1S4_2_####.cbf:1:30").strpath

    # Test screen19 first.
    with tmpdir.as_cwd():
        Screen19().run([data_dir], set_up_logging=True)

    logfile = tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile
    assert "photon incidence rate is outside the linear response region" in logfile

    # Then check screen19.minimum_exposure.
    with tmpdir.as_cwd():
        minimum_exposure.run(
            args=["integrated.expt", "integrated.refl"], set_up_logging=True
        )

    m_e_logfile = tmpdir.join("screen19.minimum_exposure.log").read()

    assert (
        "You can achieve your desired exposure factor by modifying transmission "
        "and/or exposure time." in m_e_logfile
    )


def test_screen19_single_frame(dials_data, tmpdir):
    image = dials_data("x4wide").join("X4_wide_M1S4_2_0001.cbf").strpath

    with tmpdir.as_cwd():
        Screen19().run([image], set_up_logging=True)

    logfile = tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile
