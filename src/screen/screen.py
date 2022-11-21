"""The main screening script."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from __init__ import (  # FIXME TODO change to relative import
    config_parser,
    version_parser,
)

import libtbx.phil

template_pattern = re.compile(r"(.*)_(?:[0-9]*\#+).(.*)")

import_scope = libtbx.phil.parse(
    """
      include scope dials.command_line.dials_import.phil_scope
    """,
    process_includes=True,
)

find_spots_scope = libtbx.phil.parse(
    """
    include scope dials.command_line.find_spots.phil_scope
  """,
    process_includes=True,
)

index_scope = libtbx.phil.parse(
    """
    include scope dials.command_line.index.phil_scope
  """,
    process_includes=True,
)

refine_scope = libtbx.phil.parse(
    """
    include scope dials.command_line.refine.phil_scope
  """,
    process_includes=True,
)

integrate_scope = libtbx.phil.parse(
    """
    include scope dials.command_line.integrate.phil_scope
  """,
    process_includes=True,
)

phil_scope = libtbx.phil.parse(
    """
    log = False
      .type = bool
    dials_import {
      include scope screen.import_scope
    }
    dials_find_spots{
      include scope screen.find_spots_scope
    }
    dials_index {
      include scope screen.index_scope
    }
    dials_refine {
      include scope screen.refine_scope
    }
    dials_integrate {
      include scope screen.integrate_scope
    }
    """,
    process_includes=True,
)


class _ImportImages(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 1:  # ie. already a list
            expt, directory, template = (values, None, None)
        else:
            expt, directory, template = self.find_import_arguments(values)
        setattr(namespace, self.dest, expt)
        setattr(namespace, "directory", directory)
        setattr(namespace, "template", template)

    @staticmethod
    def find_import_arguments(val) -> tuple[str | None]:  # expts, dir, template
        in_value = Path(val[0]).expanduser().resolve()
        match = template_pattern.fullmatch(in_value.stem)
        if in_value.is_dir() is True:
            return None, in_value.as_posix(), None
        elif match:
            return None, None, in_value.as_posix()
        else:
            return in_value.as_posix(), None, None


parser = argparse.ArgumentParser(
    description=__doc__, parents=[version_parser, config_parser]
)
parser.add_argument(
    "experiments", type=str, nargs="+", action=_ImportImages, help=""
)  # FIXME TODO add file.cbf:1:100
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


def run_find_spots(params):
    find_spots_params = import_scope.format(python_object=params)

    subprocess.run(["dials.find_spots", "imported.expt", find_spots_params.as_str()])


def run_indexing(params):
    index_params = import_scope.format(python_object=params)

    subprocess.run(
        ["dials.index", "imported.expt", "strong.refl", index_params.as_str()]
    )


def run_refine(params):
    refine_params = import_scope.format(python_object=params)

    subprocess.run(
        ["dials.refine", "indexed.expt", "indexed.refl", refine_params.as_str()]
    )


def run_integrate(params):
    integrate_params = import_scope.format(python_object=params)

    subprocess.run(
        ["dials.integrate", "refined.expt", "refined.refl", integrate_params.as_str()]
    )


def run_minimum_exposure():
    # subprocess.run(["screen19.minimum_exposure", "integrated.expt", "integrated.refl"])
    pass


def pipeline(args, working_phil):
    params = working_phil.extract()

    # Set directory/template if that's what's been parsed.
    params.dials_import.input.directory = [args.directory] if args.directory else []
    params.dials_import.input.template = [args.template] if args.template else []

    run_import(args.experiments, params.dials_import)
    # FIXME at the moment after dials_import it doesn't seem to correctly read input phil parameters ...
    run_find_spots(params.dials_find_spots)
    run_indexing(params.dials_index)
    subprocess.run(["dev.dials.pixel_histogram", "indexed.refl"])
    run_refine(params.dials_refine)
    run_integrate(params.dials_integrate)
    run_minimum_exposure()


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
