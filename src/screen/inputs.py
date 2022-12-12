import argparse

import libtbx.phil

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

options_parser = argparse.ArgumentParser(add_help=False)
options_parser.add_argument(
    "--find-spots",
    type=str,
    default=[],
    nargs="*",
    help="Additional options for spot finding.",
)
options_parser.add_argument(
    "--index", type=str, default=[], nargs="*", help="Additional options for indexing."
)
options_parser.add_argument(
    "--refine",
    type=str,
    default=[],
    nargs="*",
    help="Additional options for refinement.",
)
options_parser.add_argument(
    "--integrate",
    type=str,
    default=[],
    nargs="*",
    help="Additional options for integration.",
)
