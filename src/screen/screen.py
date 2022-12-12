"""Pipeline to process screening data obtained at Diamond Light Source Beamline I19."""

from __future__ import annotations

help_message = """
This program presents the user with recommendations for adjustments to beam
flux, based on a single-sweep screening data collection.  It presents an
upper- and lower-bound estimate of suitable flux.\n
  • The upper-bound estimate is based on a comparison of a histogram of
  measured pixel intensities with the trusted intensity range of the detector.
  The user is warned when the measured pixel intensities indicate that the
  detector would have a significant number of overloaded or untrustworthy
  pixels.\n
  • The lower-bound estimate is based on a linear fit of isotropic disorder
  parameter, B, to a Wilson plot of reflection intensities.  From this, an estimate
  is made of the minimum exposure (flux × exposure time) required to achieve a target
  I/σ ratio (by default, target I/σ = 2) at one or more values of desired resolution, d,
  (by default, desired d = 1 Å, 0.84 Å, 0.6 Å & 0.4 Å).\n
\n
Target I/σ and target d (in Ångström) can be set using the parameters
'min_i_over_sigma' and 'desired_d'.  One can set multiple values of the latter.\n
\n
By default the disorder parameter fit is conducted on the integrated data.
This ought to provide a reasonably true fit, but requires an integration step,
which can take some time. You can achieve a quicker, dirtier answer by fitting
to the indexed data (i.e. only the stronger spots), using 'minimum_exposure.data=indexed'.\n
\n
Examples:\n

  screen19 *.cbf\n

  screen19 /path/to/data/\n

  screen19 /path/to/data/image0001.cbf:1:100\n

  screen19 <image_files> -d indexed\n

  screen19 <image_files> --find-spots nproc=4 --index space_group=P222
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import libtbx.phil

from screen import config_parser, version_parser  # FIXME TODO change to relative import
from screen.inputs import (
    find_spots_scope,
    import_scope,
    index_scope,
    integrate_scope,
    options_parser,
    refine_scope,
)

# from screen.minimum_exposure import phil_scope as minimum_exposure_scope

# Custom types
Scope = libtbx.phil.scope
ScopeExtract = libtbx.phil.scope_extract

template_pattern = re.compile(r"(.*)_(?:[0-9]*\#+).(.*)")


phil_scope = libtbx.phil.parse(
    """
    verbosity = 0
      .type = int(value_min=0)
      .help = "Verbosity level of log output. Possible values:\n"
        "\t• 0: Info log output to stdout/logfile\n"
        "\t• 1: Info log output to stdout/logfile, logfile contains timing"
        "information\n"
        "\t• 2: Info & debug log output to stdout/logfile, logfile contains"
            "timing information"
    output {
      log = screen19.log
        .type = str
        .help = The log file name.
    }
    dials_import {
      include scope screen.inputs.import_scope
    }
    dials_find_spots{
      include scope screen.inputs.find_spots_scope
    }
    dials_index {
      include scope screen.inputs.index_scope
    }
    dials_refine {
      include scope screen.inputs.refine_scope
    }
    dials_integrate {
      include scope screen.inputs.integrate_scope
    }
    include scope screen.minimum_exposure.phil_scope
    """,
    process_includes=True,
)


class _ImportImages(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        expt, directory, template, image_range = self.find_import_arguments(values)
        setattr(namespace, self.dest, expt)
        setattr(namespace, "directory", directory)
        setattr(namespace, "template", template)
        setattr(namespace, "image_range", image_range)

    @staticmethod
    def find_import_arguments(
        val,
    ) -> tuple[str | None]:  # expts, dir, template, image_range
        if len(val) > 1:
            return val, None, None, None

        in_value = Path(val[0]).expanduser().resolve()
        match = template_pattern.fullmatch(in_value.stem)
        if in_value.is_dir() is True:
            return None, in_value.as_posix(), None, None
        elif match:
            return None, None, in_value.as_posix(), None
        elif ":" in in_value.as_posix():
            if len(in_value.name.split(":")) != 3:
                raise OSError(5, "Please specify both start and end of image range.")
            filename, start, end = in_value.name.split(":")
            temp = re.split(r"([0-9#]+)(?=\.\w)", filename)[1]
            template = in_value.parent / filename.replace(temp, "#" * len(temp))
            image_range = (int(start), int(end))
            return None, None, template.as_posix(), image_range
        else:
            return in_value.as_posix(), None, None, None


usage = "%(prog)s [options] /path/to/data"

parser = argparse.ArgumentParser(
    usage=usage,
    formatter_class=argparse.RawTextHelpFormatter,
    description=__doc__,
    epilog=help_message,
    parents=[version_parser, config_parser, options_parser],
)
parser.add_argument(
    "experiments",
    type=str,
    nargs="+",
    action=_ImportImages,
    help="The experiment path - either a directory or a list of files.",
)
parser.add_argument(
    "phil_args", nargs="*", help="Phil parameters for pipeline."
)  # I think at this point this might only be needed for import and minimum_exposure???
parser.add_argument(
    "-d",
    "--data",
    type=str,
    choices=["indexed", "integrated"],
    default="integrated",
    help="At which point the disorder parameter fit should be conducted. Defaults to 'integrated'.",
)


def run_import(images: list | str | None, params: ScopeExtract):
    # Ugly, but works
    import_params = import_scope.format(python_object=params)

    if images and type(images) is str:
        subprocess.run(["dials.import", images, import_params.as_str()])
    elif images and type(images) is list:
        subprocess.run(["dials.import", *images, import_params.as_str()])
    else:
        subprocess.run(["dials.import", import_params.as_str()])


def run_find_spots(params: ScopeExtract, options: list = []):
    find_spots_params = find_spots_scope.format(python_object=params)

    subprocess.run(
        ["dials.find_spots", "imported.expt", find_spots_params.as_str(), *options]
    )


def run_indexing(params: ScopeExtract, options: list = []):
    index_params = index_scope.format(python_object=params)

    subprocess.run(
        ["dials.index", "imported.expt", "strong.refl", index_params.as_str(), *options]
    )


def run_refine(params: ScopeExtract, options: list = []):
    refine_params = refine_scope.format(python_object=params)

    subprocess.run(
        [
            "dials.refine",
            "indexed.expt",
            "indexed.refl",
            refine_params.as_str(),
            *options,
        ]
    )


def run_integrate(params: ScopeExtract, options: list = []):
    integrate_params = integrate_scope.format(python_object=params)

    subprocess.run(
        [
            "dials.integrate",
            "refined.expt",
            "refined.refl",
            integrate_params.as_str(),
            *options,
        ]
    )


def run_minimum_exposure(choice: str, options: list = []):
    if choice == "indexed":
        subprocess.run(
            [
                "screen19.minimum_exposure",
                "indexed.expt",
                "indexed.refl",
                f"minimum_exposure.data={choice}",
                *options,
            ]
        )
    else:
        subprocess.run(
            [
                "screen19.minimum_exposure",
                "integrated.expt",
                "integrated.refl",
                f"minimum_exposure.data={choice}",
                *options,
            ]
        )


def pipeline(args: argparse.Namespace, working_phil: Scope):
    params = working_phil.extract()

    # Set directory/template if that's what's been parsed.
    params.dials_import.input.directory = [args.directory] if args.directory else []
    params.dials_import.input.template = [args.template] if args.template else []
    params.dials_import.geometry.scan.image_range = (
        args.image_range if args.image_range else None
    )

    # This probably makes the include scope in phil_scope a bit redundant... 2 choices to do this I guess?
    spot_finding_options = args.find_spots
    indexing_options = args.index
    refinement_options = args.refine
    integration_options = args.integrate
    min_exp_options = args.min_exposure

    run_import(args.experiments, params.dials_import)
    run_find_spots(params.dials_find_spots, spot_finding_options)
    run_indexing(params.dials_index, indexing_options)
    subprocess.run(["dev.dials.pixel_histogram", "indexed.refl"])
    if args.data == "integrated":
        run_refine(params.dials_refine, refinement_options)
        run_integrate(params.dials_integrate, integration_options)
        run_minimum_exposure(args.data, min_exp_options)
    else:
        run_minimum_exposure(args.data, min_exp_options)


def main(args=None):
    args = parser.parse_args(args)
    cl = phil_scope.command_line_argument_interpreter()
    working_phil = phil_scope.fetch(cl.process_and_fetch(args.phil_args))

    if args.show_config:
        # FIXME doesn't work unless some words are passed as positional argument (experiments)
        working_phil.show(attributes_level=args.attributes_level)
        sys.exit()

    pipeline(args, working_phil)


# if __name__ == "__main__":
#     main(None)
