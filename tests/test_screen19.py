# coding: utf-8

from __future__ import absolute_import, division, print_function

import shutil

import pytest

from screen19 import minimum_exposure
from screen19.screen import Screen19


def image(n):
    return "x3_1_{:04d}.cbf.gz".format(n)


# A list of tuples of example sys.argv[1:] cases and associated image count.
import_checks = [
    ([""], 900),
    (["/"], 900),
    (["x3_1_00*.cbf.gz"], 99),
    (["x3_1_####.cbf.gz:1:99"], 99),
    (["x3_1_00##.cbf.gz:1:99"], 99),
    (["x3_1_0001.cbf.gz:1:99"], 99),
    ([image(i + 1) for i in range(5)], 5),
]

# A list of tuples of example sys.argv[1:] cases, with files numbered from zero, and
# associated image count.
import_checks_zero = [
    (["x3_1_000*.cbf.gz"], 10),
    (["x3_1_####.cbf.gz:0:9"], 10),
    (["x3_1_00##.cbf.gz:0:9"], 10),
    (["x3_1_0000.cbf.gz:0:9"], 10),
    ([image(i) for i in range(5)], 5),
]


def import_data(data, image_count):  # type: (str, int) -> None
    """
    Generate and verify an imageset from spoof command-line input.

    Check that importing data according to an input string corresponding to a single
    contiguous image range results in a single imageset containing the correct number
    of images.

    Args:
        data: Valid input, such as "x3_1_####.cbf.gz", "x3_1_0001.cbf.gz:1:99", etc.
        image_count: Number of images matching the input.

    Raises:
        AssertionError: Either if more than one imageset is created or if the
                        imageset contains the wrong number of files.
    """
    screen = Screen19()
    screen._import(data)

    # Check that the import has resulted in the creation of a single experiment.
    assert len(screen.expts) == 1
    # Check that the associated imageset has the expected number of images.
    assert screen.expts[0].imageset.size() == image_count


def test_screen19_command_line_help_does_not_crash():
    Screen19().run([])


def test_minimum_exposure_help_does_not_crash():
    minimum_exposure.run(args=[])


@pytest.mark.parametrize("import_check", import_checks)
def test_screen19_inputs(dials_data, tmpdir, import_check):
    """Test various valid input argument styles"""
    data_files, image_count = import_check
    data = [dials_data("small_molecule_example") / filename for filename in data_files]

    import_data(data, image_count)


@pytest.mark.parametrize("import_check_zero", import_checks_zero)
def test_screen19_inputs_zero(dials_data, tmpdir, import_check_zero):
    """Test various valid input argument styles with filenames numbered from zero."""
    data_files, image_count = import_check_zero
    with tmpdir.as_cwd():
        # Copy x3_1_0001.cbf.gz to <tmpdir>/x3_1_0000.cbf.gz, etc.
        for i in range(10):
            shutil.copy(dials_data("small_molecule_example") / image(i + 1), image(i))

    data = [tmpdir / filename for filename in data_files]

    import_data(data, image_count)


def test_screen19(dials_data, tmpdir):
    """An integration test.  Check the full functionality of screen19."""
    data_dir = dials_data("x4wide") / "X4_wide_M1S4_2_####.cbf:1:30"

    # Test screen19 first.
    with tmpdir.as_cwd():
        Screen19().run([data_dir.strpath], set_up_logging=True)

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
    single_image = dials_data("x4wide") / "X4_wide_M1S4_2_0001.cbf"

    with tmpdir.as_cwd():
        Screen19().run([single_image.strpath], set_up_logging=True)

    logfile = tmpdir.join("screen19.log").read()

    assert "screen19 successfully completed" in logfile
