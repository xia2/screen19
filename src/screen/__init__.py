"""Screening program for small-molecule single-crystal X-ray diffraction data"""
from __future__ import annotations

__author__ = "Diamond Light Source - Scientific Software"
__email__ = "data_analysis@diamond.ac.uk"
__version__ = "0.213"
__version_tuple__ = tuple(int(x) for x in __version__.split("."))


import argparse
import logging
import re
import subprocess
import sys
import traceback

from dials.util import version

logger = logging.getLogger("dials.screen19")
debug, info, warn = logger.debug, logger.info, logger.warning

version_parser = argparse.ArgumentParser(add_help=False)
version_parser.add_argument(
    "-V",
    "--version",
    action="version",
    version=f"Screen19 version {__version__}, using DIALS {version.dials_version()}.",
)

config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument(
    "-c",
    "--show-config",
    action="store_true",
    default=False,
    dest="show_config",
    help="Show the configuration parameters.",
)
config_parser.add_argument(
    "-a",
    "--attributes-level",
    default=0,
    type=int,
    dest="attributes_level",
    help="Set the attributes level for showing the configuration parameters.",
)

# Set axis tick positions manually. Accounts for reciprocal(-square) d-scaling
d_ticks = [5, 3, 2, 1.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]


def terminal_size() -> tuple[int, int]:
    """Find the current size of the terminal window.

    :return: Tuple[int, int]: Number of columns; number of rows
    """
    columns, rows = (80, 25)
    if sys.stdout.isatty():
        try:
            result = subprocess.run(["stty", "size"], timeout=1, capture_output=True)
            rows, columns = (int(i) for i in result.stdout.decode("utf-8").split())
        except Exception as err:
            logger.exception(err)
            logger.warning("Exception caught and ignored, using default size.")
            pass
    columns = min(columns, 120)
    rows = min(rows, columns // 3)

    return columns, rows


def prettyprint_dictionary(d):
    """
    Produce a nice string representation of a dictionary, for printing.

    :param d: Dictionary to be printed.
    :type d: Dict[Optional[Any]]
    :return: String representation of :param d:.
    :rtype: str
    """
    return "{\n%s\n}" % "\n".join(
        "  %s: %s"
        % (
            k,
            str(v.decode("latin-1") if isinstance(v, bytes) else v).replace(
                "\n", "\n%s" % (" " * (4 + len(k)))
            ),
        )
        for k, v in d.items()
    )


def prettyprint_process(d):
    """
    Produce a nice string representation of a subprocess CompletedProcess object, for printing.

    :param d: ReturnObject to be printed.
    :return: String representation of :param d:.
    :rtype: str
    """
    return prettyprint_dictionary(
        {
            "command": d.args,
            "exitcode": d.returncode,
            "stdout": d.stdout,
            "stderr": d.stderr,
        }
    )


def plot_intensities(
    bins,
    hist_value_factor,
    title="'Pixel intensity distribution'",
    xlabel="'% of maximum'",
    ylabel="'Number of pixels'",
    xticks="",
    style="with boxes",
):
    """
    Create an ASCII art histogram of intensities.

    :param bins:
    :param hist_value_factor:
    :param title:
    :param xlabel:
    :param ylabel:
    :param xticks:
    :param style:
    """
    columns, rows = terminal_size()

    command = ["gnuplot"]
    plot_commands = [
        "set term dumb %d %d" % (columns, rows - 2),
        "set title %s" % title,
        "set xlabel %s" % xlabel,
        "set ylabel %s offset character %d,0" % (ylabel, len(ylabel) // 2),
        "set logscale y",
        "set boxwidth %f" % hist_value_factor,
        "set xtics %s out nomirror" % xticks,
        "set ytics out",
        "plot '-' using 1:2 title '' %s" % style,
    ]
    for x in sorted(bins.keys()):
        plot_commands.append("%f %d" % (x * hist_value_factor, bins[x]))
    plot_commands.append("e")

    debug("running %s with:\n  %s\n", " ".join(command), "\n  ".join(plot_commands))

    try:
        result = subprocess.run(
            command,
            input="\n".join(plot_commands).encode("utf-8") + b"\n",
            timeout=120,
            capture_output=True,
            env={"LD_LIBRARY_PATH": ""},
        )
    except (OSError, subprocess.TimeoutExpired):
        info(traceback.format_exc())
        warn(
            "Error running gnuplot. Cannot plot intensity distribution.  "
            "No exit code."
        )
        return
    else:
        debug("result = %s", prettyprint_process(result))

    returncode = getattr(result, "returncode")
    if returncode:
        warn(
            "Error running gnuplot. Cannot plot intensity distribution. "
            "Exit code %d",
            returncode,
        )
    else:
        star = re.compile(r"\*")
        state = set()
        for line in result.stdout.decode("utf-8").split("\n"):
            if line.strip() != "":
                stars = {m.start(0) for m in re.finditer(star, line)}
                if not stars:
                    state = set()
                else:
                    state |= stars
                    line = list(line)
                    for s in state:
                        line[s] = "*"
                info("".join(line))
