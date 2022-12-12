"""The main screening script."""
from __future__ import annotations

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

template_pattern = re.compile(r"(.*)_(?:[0-9]*\#+).(.*)")


phil_scope = libtbx.phil.parse(
    """
    log = False
      .type = bool
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


parser = argparse.ArgumentParser(
    description=__doc__, parents=[version_parser, config_parser, options_parser]
)
parser.add_argument("experiments", type=str, nargs="+", action=_ImportImages, help="")
parser.add_argument("phil_args", nargs="*")


def run_import(images, params):
    # Ugly, but works
    import_params = import_scope.format(python_object=params)

    if images and type(images) is str:
        subprocess.run(["dials.import", images, import_params.as_str()])
    elif images and type(images) is list:
        subprocess.run(["dials.import", *images, import_params.as_str()])
    else:
        subprocess.run(["dials.import", import_params.as_str()])


def run_find_spots(params, options=[]):
    find_spots_params = find_spots_scope.format(python_object=params)

    subprocess.run(
        ["dials.find_spots", "imported.expt", find_spots_params.as_str(), *options]
    )


def run_indexing(params, options=[]):
    index_params = index_scope.format(python_object=params)

    subprocess.run(
        ["dials.index", "imported.expt", "strong.refl", index_params.as_str(), *options]
    )


def run_refine(params, options=[]):
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


def run_integrate(params, options=[]):
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


def run_minimum_exposure():
    # subprocess.run(["screen19.minimum_exposure", "integrated.expt", "integrated.refl"])
    pass


def pipeline(args, working_phil):
    params = working_phil.extract()

    # Set directory/template if that's what's been parsed.
    params.dials_import.input.directory = [args.directory] if args.directory else []
    params.dials_import.input.template = [args.template] if args.template else []
    params.dials_import.geometry.scan.image_range = (
        args.image_range if args.image_range else None
    )

    spot_finding_options = args.find_spots
    indexing_options = args.index
    refinement_options = args.refine
    integration_options = args.integrate

    run_import(args.experiments, params.dials_import)
    run_find_spots(params.dials_find_spots, spot_finding_options)
    run_indexing(params.dials_index, indexing_options)
    subprocess.run(["dev.dials.pixel_histogram", "indexed.refl"])
    run_refine(params.dials_refine, refinement_options)
    run_integrate(params.dials_integrate, integration_options)
    run_minimum_exposure()  # If minimum exposure at indexing, stope there, else go on to integrate


def main(args=None):
    args = parser.parse_args(args)
    cl = phil_scope.command_line_argument_interpreter()
    working_phil = phil_scope.fetch(cl.process_and_fetch(args.phil_args))

    if args.show_config:
        # FIXME doesn't work unless some words are passed as positional arguments
        working_phil.show(attributes_level=args.attributes_level)
        sys.exit()

    pipeline(args, working_phil)


if __name__ == "__main__":
    main(None)
