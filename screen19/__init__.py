"""Common tools for the I19 module."""

import logging
import os
import re
import subprocess
import sys
import traceback
from typing import Dict, Tuple  # noqa: F401

import procrunner

# Flake8 does not detect typing yet (https://gitlab.com/pycqa/flake8/issues/342)

__version__ = "0.213"

logger = logging.getLogger("dials.screen19")
debug, info, warn = logger.debug, logger.info, logger.warning


# Set axis tick positions manually.  Accounts for reciprocal(-square) d-scaling.
d_ticks = [5, 3, 2, 1.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]


def terminal_size() -> Tuple[int, int]:
    """
    Find the current size of the terminal window.

    :return: Number of columns; number of rows.
    """
    columns, rows = 80, 25
    if sys.stdout.isatty():
        try:
            result = procrunner.run(
                ["stty", "size"],
                timeout=1,
                raise_timeout_exception=True,
                print_stdout=False,
                print_stderr=False,
            )
            rows, columns = (int(i) for i in result.stdout.decode("utf-8").split())
        except Exception:  # ignore any errors and use default size
            pass  # FIXME: Can we be more specific about the type of exception?
    columns = min(columns, 120)
    rows = min(rows, int(columns / 3))

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


def prettyprint_procrunner(d):
    """
    Produce a nice string representation of a procrunner ReturnObject, for printing.

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


def make_template(f):
    """
    Generate a xia2-style filename template.

    From a given filename, generate a template filename by substituting a hash
    character (#) for each numeral in the last contiguous group of numerals
    before the file extension.
    For example, the filename example_01_0001.cbf becomes example_01_####.cbf.

    :param f: Filename, with extension.
    :type f: str
    :return: Filename template, with extension; image number.
    :rtype: Tuple(str, int)
    """
    # Split the file from its path
    directory, f = os.path.split(f)
    # Split off the file extension, assuming it begins at the first full stop,
    # also split the last contiguous group of digits off the filename root
    parts = re.split(r"([0-9#]+)(?=\.\w)", f, 1)
    # Get the number of digits in the group we just isolated and their value
    try:
        # Combine the root, a hash for each digit and the extension
        length = len(parts[1])
        template = parts[0] + "#" * length + parts[2]
        image = int(parts[1].replace("#", "0"))
    except IndexError:
        template = parts[0]
        image = None
    return os.path.join(directory, template), image


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
        result = procrunner.run(
            command,
            stdin="\n".join(plot_commands).encode("utf-8") + b"\n",
            timeout=120,
            raise_timeout_exception=True,
            print_stdout=False,
            print_stderr=False,
            environment_override={"LD_LIBRARY_PATH": ""},
        )
    except (OSError, subprocess.TimeoutExpired):
        info(traceback.format_exc())
        warn(
            "Error running gnuplot. Cannot plot intensity distribution.  "
            "No exit code."
        )
        return
    else:
        debug("result = %s", prettyprint_procrunner(result))

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
