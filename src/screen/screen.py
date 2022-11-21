"""The main screening script."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

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


parser = argparse.ArgumentParser(description=__doc__)
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


def pipeline(args, working_phil):
    params = working_phil.extract()

    # Set directory/template if that's what's been parsed.
    params.dials_import.input.directory = [args.directory] if args.directory else []
    params.dials_import.input.template = [args.template] if args.template else []

    run_import(args.experiments, params.dials_import)
    run_find_spots(params.dials_find_spots)
    run_indexing(params.dials_index)
    subprocess.run(["dev.dials.pixel_histogram", "indexed.refl"])


# def pipeline(images):
# subprocess.run(["dials.refine", "indexed.expt", "indexed.refl"])
# subprocess.run(["dials.integrate", "refined.expt", "refined.refl"])
# subprocess.run(["screen19.minimum_exposure", "integrated.expt", "integrated.refl"])


def main(args=None):
    args = parser.parse_args(args)
    cl = phil_scope.command_line_argument_interpreter()
    working_phil = phil_scope.fetch(cl.process_and_fetch(args.phil_args))

    pipeline(args, working_phil)


if __name__ == "__main__":
    main(None)
