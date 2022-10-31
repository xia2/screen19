import pytest

import dxtbx

from screen import minimum_exposure, screen

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


def test_screen19_help_does_not_crash():
    screen.main(args=[None])


def test_minimum_exposure_help_does_not_crash():
    minimum_exposure.run(args=[])


@pytest.mark.parametrize("import_checks", import_checks)
def test_screen19_inputs(dials_data, tmp_path, import_checks):
    """Test various valid input argument styles"""
    data_files, image_count = import_checks
    data = [
        dials_data("small_molecule_example").join(filename).strpath
        for filename in data_files
    ]

    # Run screen19
    screen.main(data)

    # Check that the import has resulted in the creation of a single experiment.
    expts = dxtbx.model.ExperimentList.from_files(tmp_path / "imported.expt")
    assert len(expts) == 1
    # Check that the associated imageset has the expected number of images.
    assert expts[0].imageset.size() == image_count
