#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""
Process screening data obtained at Diamond Light Source Beamline I19.

This program presents the user with recommendations for adjustments to beam
flux, based on a single-sweep screening data collection.  It presents an
upper- and lower-bound estimate of suitable flux.
  * The upper-bound estimate is based on a comparison of a histogram of
  measured pixel intensities with the trusted intensity range of the detector.
  The user is warned when the measured pixel intensities indicate that the
  detector would have a significant number of overloaded or untrustworthy
  pixels.
  * The lower-bound estimate is based on a linear fit of isotropic disorder
  parameter, B, to a Wilson plot of reflection intensities.  From this,
  an estimate is made of the minimum flux required to achieve a target I/σ
  ratio (by default, target I/σ = 2) at one or more values of desired
  resolution, d, (by default, desired d = 1 Å, 0.84 Å, 0.6 Å & 0.4 Å).

Target I/σ and target d (in Ångström) can be set using the parameters
'min_i_over_sigma' and 'desired_d'.  One can set multiple values of the latter.

By default, for speed, the disorder parameter fit is conducted on the
indexed data (i.e. only the strong spots).  This may not provide a good
estimate in some cases.  If you suspect the fit is poor, try using the
integrated data instead, using 'lower_bound_estimate.data=integrated'

Examples:

  i19.screen datablock.json

  i19.screen *.cbf

  i19.screen /path/to/data/

  i19.screen /path/to/data/image0001.cbf:1:100

  i19.screen min_i_over_sigma=2 desired_d=0.84 <datablock.json | image_files>

  i19.screen lower_bound_estimate.data=integrated <image_files>

"""

from __future__ import absolute_import, division, print_function

import json
import logging

from typing import Dict, List, Tuple, Optional

import math
import os
import re
import sys
import time
import timeit
import traceback

import procrunner
import iotbx.phil
from libtbx import easy_pickle
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.util.options import OptionParser


help_message = __doc__

phil_scope = iotbx.phil.parse(
    """
nproc = None
  .type = int
  .caption = 'Number of processors to use'
  .help = 'The chosen value will apply to all the DIALS utilities with a ' \
          "multi-processing option.  If 'None' is given, all available " \
          'processors will be used.'

lower_bound_estimate
    .caption = 'Parameters for the calculation of the lower flux bound'
    {
    data = indexed *integrated
        .type = choice
        .caption = 'Choice of data for the displacement parameter fit'
        .help = 'For the lower-bound flux estimate, choose whether to use ' \
                'indexed (quicker) or integrated (better) data in fitting ' \
                'the isotropic displacement parameter.'
    desired_d = None
        .multiple = True
        .type = float
        .caption = u'Desired resolution limit, in Ångströms, of diffraction ' \
                   u'data'
        .help = 'This is the resolution target for the lower-bound flux ' \
                'recommendation.'
    min_i_over_sigma = 2
        .type = float
        .caption = u'Target I/σ value for lower-bound flux recommendation'
        .help = u'The lower-bound flux recommendation provides an estimate ' \
                u'of the flux required to ensure that the majority of' \
                u'expected reflections at the desired resolution limit have ' \
                u'I/σ greater than or equal to this value.'
    wilson_fit_max_d = 4  # Å
        .type = float
        .caption = u'Maximum d-value (in Ångströms) for displacement ' \
                   u'parameter fit'
        .help = 'Reflections with lower resolution than this value will be ' \
                'ignored for the purposes of the Wilson plot.'
    }

!dials_import
  .caption = 'Options for dials.import'
  {
  include scope dials.command_line.dials_import.phil_scope
  }

!dials_find_spots
  .caption = 'Options for dials.find_spots'
  {
  include scope dials.command_line.find_spots.phil_scope
  }

!dials_index
  .caption = 'Options for dials.index'
  {
  include scope dials.command_line.index.phil_scope
  }

!dials_refine
  .caption = 'Options for dials.refine'
  {
  include scope dials.command_line.refine.phil_scope
  }

!dials_refine_bravais
  .caption = 'Options for dials.refine_bravais_settings'
  {
  include scope dials.command_line.refine_bravais_settings.phil_scope
  }

!dials_create_profile
  .caption = 'Options for dials.create_profile_model'
  {
  include scope dials.command_line.create_profile_model.phil_scope
  }

!dials_integrate
  .caption = 'Options for dials.integrate'
  {
  include scope dials.command_line.integrate.phil_scope
  }

!dials_report
  .caption = 'Options for dials.report'
  {
  include scope dials.command_line.report.phil_scope
  }
""",
    process_includes=True,
)

procrunner_debug = False
logger = logging.getLogger("dials.i19.screen")
debug, info, warn = logger.debug, logger.info, logger.warn


def terminal_size():
    """
    Find the current size of the terminal window.

    :return: Number of columns; number of rows.
    :rtype: Tuple[int]
    """
    columns, rows = 80, 25
    if sys.stdout.isatty():
        try:
            result = procrunner.run(
                ["stty", "size"],
                timeout=1,
                print_stdout=False,
                print_stderr=False,
                debug=procrunner_debug,
            )
            rows, columns = [int(i) for i in result["stdout"].split()]
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
        [
            "  %s: %s" % (k, str(v).replace("\n", "\n%s" % (" " * (4 + len(k)))))
            for k, v in d.items()
        ]
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
    :return: Filename template, with extension.
    :rtype: str
    """
    # Split the file from its path
    directory, f = os.path.split(f)
    # Split off the file extension, assuming it begins at the first full stop,
    # also split the last contiguous group of digits off the filename root
    parts = re.split("([0-9]+)([\.\w+]+)$", f, 1)
    # Get the number of digits in the group we just isolated and their value
    try:
        # Combine the root, a hash for each digit and the extension
        length = len(parts[1])
        template = parts[0] + "#" * length + parts[2]
        image = int(parts[1])
    except IndexError:
        template = parts[0]
        image = None
    return os.path.join(directory, template), image


class I19Screen(object):
    """
    Encapsulates the screening script.
    """

    # TODO Make __init__ and declare instance variables in it.
    def _quick_import(self, files):
        """
    TODO: Docstring
    :param files:
    :type files: List[str]
    :return:
    """
        if len(files) == 1:
            # No point in quick-importing a single file
            return False
        debug("Attempting quick import...")
        files.sort()
        templates = {}  # type: Dict[str, List[Optional[List[int]]]]
        for f in files:
            template, image = make_template(f)
            if template not in templates:
                image_range = [image, image] if image else []
                templates.update({template: [image_range]})
            elif image == templates[template][-1][-1] + 1:
                templates[template][-1][-1] = image
            else:
                templates[template].append([image, image])
        # Return tuple of template and image range for each unique image range
        templates = [(t, tuple(r)) for t, ranges in templates.items()
                     for r in ranges]
        # type: List[Tuple[str, Tuple[int]]]
        return self._quick_import_templates(templates)

    def _quick_import_templates(self, templates):
        """
        TODO: Docstring
        :param templates:
        :return:
        """
        debug("Quick import template summary:")
        debug(templates)
        if len(templates) > 1:
            debug("Cannot currently run quick import on multiple templates")
            return False

        try:
            scan_range = templates[0][1]  # type: Tuple[int]
            if not scan_range:
                raise IndexError
        except IndexError:
            debug("Cannot run quick import: could not determine image naming "
                  "template")
            return False

        info("Running quick import")
        self._run_dials_import(
            [
                "input.template=%s" % templates[0][0],
                "geometry.scan.image_range=%d,%d" % scan_range,
                "geometry.scan.extrapolate_scan=True",
            ]
        )
        return True

    def _import(self, files):
        """
        TODO: Docstring
        :param files:
        :return:
        """
        info("\nImporting data...")
        if len(files) == 1:
            if os.path.isdir(files[0]):
                debug(
                    "You specified a directory. Importing all CBF files in "
                    "that directory."
                )
                # TODO Support other image formats for more general application
                files = [
                    os.path.join(files[0], f)
                    for f in os.listdir(files[0])
                    if f.endswith(".cbf")
                ]
            elif len(files[0].split(":")) == 3:
                debug(
                    "You specified an image range in the xia2 format.  "
                    "Importing all specified files."
                )
                template, start, end = files[0].split(":")
                template = make_template(template)[0]
                start, end = int(start), int(end)
                if not self._quick_import_templates(
                    [(template, (start, end))]
                ):
                    warn("Could not import specified image range.")
                    sys.exit(1)
                info("Quick import successful")
                return

        # Can the files be quick-imported?
        if self._quick_import(files):
            info("Quick import successful")
            return

        self._run_dials_import(files)

    def _run_dials_import(self, parameters):
        """
        TODO: Docstring
        :param parameters:
        :return:
        """
        command = ["dials.import"] + parameters
        # + ['allow_multiple_sweeps=true']
        debug("running %s" % " ".join(command))

        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))

        if result["exitcode"] == 0:
            if os.path.isfile("datablock.json"):
                info("Successfully completed (%.1f sec)" % result["runtime"])
            else:
                warn(
                    "Could not import images. Do the specified images exist "
                    "at that location?"
                )
                sys.exit(1)
        else:
            if "More than 1 sweep was found." in result["stderr"]:
                warn(
                    "The data contain multiple sweeps. i19.screen can only "
                    "run on a single sweep of data."
                )
                sys.exit(1)
            warn("Failed with exit code %d" % result["exitcode"])
            sys.exit(1)

    def _count_processors(self, nproc=None):
        """
        Determine the number of processors and save it as an instance variable.

        The user may specify the number of processors to use.  If no value is
        given, the number of available processors is returned.
        TODO: Once we're using the proper option parser, this becomes redundant?

        :param nproc: User-specified number of processors to use.
        :type nproc: int
        """
        if nproc is not None:
            self.nproc = nproc
            return

        # if environmental variable NSLOTS is set to a number then use that
        try:
            self.nproc = int(os.environ.get("NSLOTS"))
            return
        except (ValueError, TypeError):
            pass

        from libtbx.introspection import number_of_processors

        self.nproc = number_of_processors(return_value_if_unknown=-1)

        if self.nproc <= 0:
            warn(
                "Could not determine number of available processors. "
                "Error code %d"
                "" % self.nproc
            )
            sys.exit(1)

    def _count_images(self):
        """
        Attempt to determine the number of diffraction images.

        The number of diffraction images is determined from the datablock JSON
        file.

        :return: Number of images.
        :rtype: int
        """
        with open(self.json_file) as fh:
            datablock = json.load(fh)
        try:
            return sum(len(s["exposure_time"]) for s in datablock[0]["scan"])
        except Exception:  # FIXME: Can we be specific?
            warn("Could not determine number of images in dataset")
            sys.exit(1)

    def _check_intensities(self, mosaicity_correction=True):
        """
        TODO: Docstring
        :param mosaicity_correction:
        :return:
        """
        info("\nTesting pixel intensities...")
        command = ["xia2.overload", "nproc=%s" % self.nproc, self.json_file]
        debug("running %s" % command)
        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))
        info("Successfully completed (%.1f sec)" % result["runtime"])

        if result["exitcode"] != 0:
            warn("Failed with exit code %d" % result["exitcode"])
            sys.exit(1)

        with open("overload.json") as fh:
            overload_data = json.load(fh)

        print("Pixel intensity distribution:")
        count_sum = 0
        hist = {}
        if "bins" in overload_data:
            for b in range(overload_data["bin_count"]):
                if overload_data["bins"][b] > 0:
                    hist[b] = overload_data["bins"][b]
                    count_sum += b * overload_data["bins"][b]
        else:
            hist = {int(k): v for k, v in overload_data["counts"].items()
                    if int(k) > 0}
            count_sum = sum([k * v for k, v in hist.items()])

        average_to_peak = 1
        if mosaicity_correction:
            # we have checked this: if sigma_m >> oscillation it works out
            # about 1 as you would expect
            if self._sigma_m:
                M = (
                    math.sqrt(math.pi)
                    * self._sigma_m
                    * math.erf(self._oscillation / (2 * self._sigma_m))
                )
                average_to_peak = M / self._oscillation
                info("Average-to-peak intensity ratio: %f" % average_to_peak)

        scale = 100 * overload_data["scale_factor"] / average_to_peak
        info("Determined scale factor for intensities as %f" % scale)

        debug(
            "intensity histogram: { %s }",
            ", ".join(["%d:%d" % (k, hist[k]) for k in sorted(hist)]),
        )
        max_count = max(hist.keys())
        hist_max = max_count * scale
        hist_granularity, hist_format = 1, "%.0f"
        if hist_max < 50:
            hist_granularity, hist_format = 2, "%.1f"
        if hist_max < 15:
            hist_granularity, hist_format = 10, "%.1f"
        rescaled_hist = {}
        for x in hist.keys():
            rescaled = round(x * scale * hist_granularity)
            if rescaled > 0:
                rescaled_hist[rescaled] = \
                    hist[x] + rescaled_hist.get(rescaled, 0)
        hist = rescaled_hist
        debug(
            "rescaled histogram: { %s }",
            ", ".join(
                [
                    (hist_format + ":%d") % (k / hist_granularity, hist[k])
                    for k in sorted(hist)
                ]
            ),
        )

        self._plot_intensities(hist, 1 / hist_granularity)

        text = "".join(
            (
                "Strongest pixel (%d counts) " % max_count,
                "reaches %.1f%% " % hist_max,
                "of the detector count rate limit",
            )
        )
        if hist_max > 100:
            warn("Warning: %s!" % text)
        else:
            info(text)
        if (
            "overload_limit" in overload_data
            and max_count >= overload_data["overload_limit"]
        ):
            warn("Warning: THE DATA CONTAIN REGULAR OVERLOADS!")
            warn(
                "         The photon incidence rate is outside the specified "
                "limits of the detector."
            )
            warn(
                "         The built-in detector count rate correction cannot "
                "adjust for this."
            )
            warn(
                "         You should aim for count rates below 25% of the "
                "detector limit."
            )
        elif hist_max > 70:
            warn(
                "Warning: The photon incidence rate is well outside the "
                "linear response region of the detector (<25%)."
            )
            warn(
                "    The built-in detector count rate correction may not be "
                "able to adjust for this."
            )
        elif hist_max > 25:
            info(
                "The photon incidence rate is outside the linear response "
                "region of the detector (<25%)."
            )
            info(
                "The built-in detector count rate correction should be able "
                "to adjust for this."
            )
        if not mosaicity_correction:
            warn("Warning: Not enough data for proper profile estimation.")
            warn("    The spot intensities are not corrected for mosaicity.")
            warn(
                "    The true photon incidence rate will be higher than the "
                "given estimate."
            )

        info("Total sum of counts in dataset: %d" % count_sum)

    @staticmethod
    def _plot_intensities(
        bins,
        hist_value_factor,
        title="'Spot intensity distribution'",
        xlabel="'% of maximum'",
        ylabel="'Number of observed pixels'",
        xticks="",
        style="with boxes",
    ):
        """
        TODO: Docstring
        :param bins:
        :param hist_value_factor:
        :param title:
        :param xlabel:
        :param ylabel:
        :param xticks:
        :param style:
        :return:
        """
        columns, rows = terminal_size()

        command = ["gnuplot"]
        plot_commands = [
            "set term dumb %d %d" % (columns, rows - 2),
            "set title %s" % title,
            "set xlabel %s" % xlabel,
            "set ylabel %s" % ylabel,
            "set logscale y",
            "set boxwidth %f" % hist_value_factor,
            "set xtics %s out nomirror" % xticks,
            "set ytics out",
            "plot '-' using 1:2 title '' %s" % style,
        ]
        for x in sorted(bins.keys()):
            plot_commands.append("%f %d" % (x * hist_value_factor, bins[x]))
        plot_commands.append("e")

        debug(
            "running %s with:\n  %s\n" %
            (" ".join(command), "\n  ".join(plot_commands))
        )

        try:
            result = procrunner.run(
                command,
                stdin="\n".join(plot_commands) + "\n",
                timeout=120,
                print_stdout=False,
                print_stderr=False,
                debug=procrunner_debug,
            )
        except OSError:
            info(traceback.format_exc())

        debug("result = %s" % prettyprint_dictionary(result))

        if result["exitcode"] == 0:
            star = re.compile(r"\*")
            state = set()
            for l in result["stdout"].split("\n"):
                if l.strip() != "":
                    stars = {m.start(0) for m in re.finditer(star, l)}
                    if not stars:
                        state = set()
                    else:
                        state |= stars
                        l = list(l)
                        for s in state:
                            l[s] = "*"
                    info("".join(l))
        else:
            warn(
                "Error running gnuplot. Cannot plot intensity distribution. "
                "Exit code %d" % result["exitcode"]
            )

    def _find_spots(self, additional_parameters=None):
        """
        TODO: Docstring
        :param additional_parameters:
        :return:
        """
        if additional_parameters is None:
            additional_parameters = []
        info("\nSpot finding...")
        command = [
            "dials.find_spots",
            self.json_file,
            "nproc=%s" % self.nproc,
        ] + additional_parameters
        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))
        if result["exitcode"] != 0:
            warn("Failed with exit code %d" % result["exitcode"])
            sys.exit(1)
        info(60 * "-")
        from dials.util.ascii_art import spot_counts_per_image_plot

        refl = easy_pickle.load("strong.pickle")
        info(spot_counts_per_image_plot(refl))
        info(60 * "-")
        info("Successfully completed (%.1f sec)" % result["runtime"])

    def _index(self):
        """
        TODO: Docstring
        :return:
        """
        base_command = [
            "dials.index",
            self.json_file,
            "strong.pickle",
            "indexing.nproc=%s" % self.nproc,
        ]
        runlist = [
            ("Indexing...", base_command),
            (
                "Retrying with max_cell constraint",
                base_command + ["max_cell=20"]
            ),
            (
                "Retrying with 1D FFT",
                base_command + ["indexing.method=fft1d"]
            ),
        ]

        for message, command in runlist:
            info("\n%s..." % message)

            result = procrunner.run(
                command, print_stdout=False, debug=procrunner_debug
            )
            debug("result = %s" % prettyprint_dictionary(result))
            if result["exitcode"] != 0:
                warn("Failed with exit code %d" % result["exitcode"])
            else:
                break

        if result["exitcode"] != 0:
            return False

        m = re.search(
            "model [0-9]+ \(([0-9]+) [^\n]*\n[^\n]*\n[^\n]*"
            "Unit cell: \(([^\n]*)\)\n[^\n]*Space group: ([^\n]*)\n",
            result["stdout"],
        )
        info(
            "Found primitive solution: %s (%s) using %s reflections"
            % (m.group(3), m.group(2), m.group(1))
        )
        info("Successfully completed (%.1f sec)" % result["runtime"])
        return True

    def _wilson_calculation(self):
        u"""
        Perform straight-line Wilson plot fit.  Draw the Wilson plot.

        Reflection d-spacings are determined from the crystal symmetry (from
        indexing) and the Miller indices of the indexed reflections.  The
        atomic displacement parameter is assumed isotropic.  Its value is
        determined from a fit to the reflection data:
          I = A * exp(-B /(2 * d^2)),
        where I is the intensity and the scale factor, A, and isotropic
        displacement parameter, B, are the fitted parameters.

        An I/σ condition for 'good' diffraction statistics is set by the
        instance variable min_i_over_sigma, and the user's desired
        resolution is set by the instance variable desired_d.  A crude
        error model is assumed, whereby σ² = I, and so the I/σ condition
        translates trivially to a threshold I.

        The value of the fitted intensity function at the desired
        resolution is compared with the threshold I.  The ratio of these
        values is used to determine a recommended flux for the full data
        collection.

        The Wilson plot of I as a function of d is drawn.
        """
        from dials.array_family import flex
        import numpy as np
        from scipy.optimize import curve_fit
        from cctbx import miller

        info('\nEstimating lower flux bound...')

        # TODO Convert to PHIL parser input

        if self.params.lower_bound_estimate.data == 'indexed':
            data = easy_pickle.load('indexed.pickle')
            flag = data.flags.indexed
            elist = ExperimentListFactory.from_json_file('experiments.json')
        elif self.params.lower_bound_estimate.data == 'integrated':
            data = easy_pickle.load('integrated.pickle')
            flag = data.flags.integrated
            elist = ExperimentListFactory.from_json_file(
                'integrated_experiments.json')
        else:
            warn('Unknown data option for lower-bound flux estimate.')
            sys.exit(1)
        data = data.select(data.get_flags(flag))
        crystal_symmetry = elist[0].crystal.get_crystal_symmetry()

        # Get d-spacings of indexed spots.
        def d_star_sq(x): return 1 / crystal_symmetry.unit_cell().d(x) ** 2
        d_star_sq = d_star_sq(data['miller_index'])
        intensity = data['intensity.sum.value']
        sigma = flex.sqrt(data['intensity.sum.variance'])

        # Parameters for the lower-bound flux estimate:
        min_i_over_sigma = self.params.lower_bound_estimate.min_i_over_sigma
        desired_d = self.params.lower_bound_estimate.desired_d
        desired_d.sort(reverse=True)
        wilson_fit_max_d = self.params.lower_bound_estimate.wilson_fit_max_d

        # Fit a simple Debye-Waller factor, assume isotropic disorder parameter
        def scaled_debye_waller(x, b, a): return a * np.exp(- b / 2 * x)
        sel = d_star_sq > 1 / wilson_fit_max_d**2
        # Using 1/σ weighting has a tendency to fit to the floor.
        wilson_fit, cov = curve_fit(scaled_debye_waller,
                                    d_star_sq.select(sel),
                                    intensity.select(sel),
                                    sigma=sigma.select(sel),
                                    bounds=(0, np.inf))
        # Use the fact that σ² = I for indexed data, so I/σ = √̅I
        desired_d_star_sq = [1 / d**2 for d in desired_d]
        recommended_factor = [
            (min_i_over_sigma**2 / scaled_debye_waller(target, *wilson_fit))
            for target in desired_d_star_sq]

        # Draw the Wilson plot, using existing functionality in cctbx.miller:
        columns, rows = terminal_size()
        n_bins = min(columns, intensity.size())
        ms = miller.set(crystal_symmetry=crystal_symmetry,
                        anomalous_flag=False, indices=data['miller_index'])
        ma = miller.array(ms, data=intensity, sigmas=sigma)
        ma.set_observation_type_xray_intensity()
        ma.setup_binner_counting_sorted(n_bins=n_bins)
        wilson = ma.wilson_plot(use_binning=True)
        # Get the relevant plot data from the miller_array:
        binned_intensity = [x if x else 0 for x in wilson.data[1:-1]]
        bins = dict(zip(wilson.binner.bin_centers(1), binned_intensity))
        # Set some tick positions manually, accounts for odd d-axis scaling:
        d_ticks = [5, 3, 2, 1.5, 1, .9, .8, .7, .6, .5]
        tick_positions = ', '.join(['"%g" %s' % (d, 1/d**2) for d in d_ticks])
        tick_positions = tick_positions.join(['(', ')'])
        # Draw the plot:
        self._plot_intensities(bins, 1,
                               title="'Wilson plot'",
                               xlabel="'d (Angstrom) (inverse-square scale)'",
                               ylabel="'I (counts)'",
                               xticks=tick_positions,
                               style='with lines')

        # TODO:  Remove block below for production:
        # Plots for debugging:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.xlabel(u'd (Å) (inverse-square scale)')
        plt.ylabel(u'Intensity (counts)')
        plt.xticks([1 / d ** 2 for d in d_ticks], ['%g' % d for d in d_ticks])
        plt.semilogy()
        plt.plot(d_star_sq, intensity, 'b.')
        plt.plot(d_star_sq,
                 scaled_debye_waller(d_star_sq, *wilson_fit),
                 'r-')
        plt.savefig('wilson_%s' % self.params.lower_bound_estimate.data)

        # Print a recommendation to the user.
        info('\nFitted isotropic displacement parameter, B = %.3g Angstrom^2'
             % wilson_fit[0])
        for target, recommendation in zip(desired_d, recommended_factor):
            if recommendation <= 1:
                info('\nIt is likely that you can achieve a resolution of %g '
                     'Angstrom using a lower flux.' % target)
            else:
                info('\nIt is likely that you need a higher flux to achieve a '
                     'resolution of %g Angstrom.' % target)
            info('The estimated minimal sufficient flux is %.3g times the '
                 'flux used for this data collection.' % recommendation)

    def _refine(self):
        """
        TODO: Docstring
        :return:
        """
        info("\nIndexing...")
        command = ["dials.refine", "experiments.json", "indexed.pickle"]
        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))
        if result["exitcode"] != 0:
            warn("Failed with exit code %d" % result["exitcode"])
            warn("Giving up.")
            sys.exit(1)

        info("Successfully refined (%.1f sec)" % result["runtime"])
        os.rename("experiments.json", "experiments.unrefined.json")
        os.rename("indexed.pickle", "indexed.unrefined.pickle")
        os.rename("refined_experiments.json", "experiments.json")
        os.rename("refined.pickle", "indexed.pickle")

    def _create_profile_model(self):
        """
        TODO: Docstring
        :return:
        """
        info("\nCreating profile model...")
        command = [
            "dials.create_profile_model", "experiments.json", "indexed.pickle"
        ]
        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))
        self._sigma_m = None
        if result["exitcode"] == 0:
            db = ExperimentListFactory.from_json_file(
                "experiments_with_profile_model.json"
            )[0]
            self._num_images = db.imageset.get_scan().get_num_images()
            self._oscillation = db.imageset.get_scan().get_oscillation()[1]
            self._sigma_m = db.profile.sigma_m()
            info(
                "%d images, %s deg. oscillation, sigma_m=%.3f"
                % (self._num_images, str(self._oscillation), self._sigma_m)
            )
            info("Successfully completed (%.1f sec)" % result["runtime"])
            return True
        warn("Failed with exit code %d" % result["exitcode"])
        return False

    def _integrate(self):
        """
        TODO: Docstring
        :return:
        """
        info("\nIntegrating...")
        command = [
            "dials.integrate",
            "experiments.json",
            "indexed.pickle",
            "integration.mp.nproc=%s" % self.nproc,
        ]
        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))

        if result["exitcode"] != 0:
            warn("Failed with exit code %d" % result["exitcode"])
            return False
        else:
            info("Successfully completed (%.1f sec)" % result["runtime"])
            return True

    def _refine_bravais(self):
        """
        TODO: Docstring
        :return:
        """
        info("\nRefining bravais settings...")
        command = [
            "dials.refine_bravais_settings",
            "experiments.json",
            "indexed.pickle",
        ]
        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))
        if result["exitcode"] == 0:
            m = re.search("---+\n[^\n]*\n---+\n(.*\n)*---+", result["stdout"])
            info(m.group(0))
            info("Successfully completed (%.1f sec)" % result["runtime"])
        else:
            warn("Failed with exit code %d" % result["exitcode"])
            sys.exit(1)

    def _report(self):
        """
        TODO: Docstring
        :return:
        """
        info("\nCreating report...")
        command = [
            "dials.report",
            "experiments_with_profile_model.json",
            "indexed.pickle",
        ]
        result = procrunner.run(
            command, print_stdout=False, debug=procrunner_debug
        )
        debug("result = %s" % prettyprint_dictionary(result))
        if result["exitcode"] == 0:
            info("Successfully completed (%.1f sec)" % result["runtime"])
        #     if sys.stdout.isatty():
        #       info("Trying to start browser")
        #       try:
        #         import subprocess
        #         d = dict(os.environ)
        #         d["LD_LIBRARY_PATH"] = ""
        #         subprocess.Popen(["xdg-open", "dials-report.html"], env=d)
        #       except Exception as e:
        #         debug("Could not open browser")
        #         debug(str(e))
        else:
            warn("Failed with exit code %d" % result["exitcode"])
            sys.exit(1)

    def run(self, args=None, phil=phil_scope):
        """
        TODO: Docstring
        :param args:
        :param phil:
        :return:
        """
        import libtbx.load_env
        from i19.util.version import i19_version
        from dials.util.version import dials_version

        usage = (
            "%s [options] image_directory | image_files.cbf | "
            "datablock.json" % libtbx.env.dispatcher_name
        )

        parser = OptionParser(
            usage=usage, epilog=help_message, phil=phil, check_format=False
        )

        self.params, options, unhandled = parser.parse_args(
            args=args,
            show_diff_phil=True,
            return_unhandled=True,
            quick_parse=True
        )

        version_information = "%s using %s (%s)" % (
            i19_version(),
            dials_version(),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        start = timeit.default_timer()

        if len(unhandled) == 0:
            print(help_message)
            print(version_information)
            return

        # Configure the logging
        from dials.util import log

        log.config(info="i19.screen.log", debug="i19.screen.debug.log")

        info(version_information)
        debug("Run with:")
        debug("%s\n%s" % (" ".join(unhandled), parser.diff_phil.as_str()))

        # If no target resolution is given, use the following defaults:
        if not self.params.lower_bound_estimate.desired_d:
            self.params.lower_bound_estimate.desired_d = [
                1,  # Å
                0.84,  # Å (IUCr publication requirement)
                0.6,  # Å
                0.4,  # Å
            ]

        self._count_processors(nproc=self.params.nproc)
        debug("Using %s processors" % self.nproc)

        if len(unhandled) == 1 and unhandled[0].endswith(".json"):
            self.json_file = unhandled[0]
        else:
            self._import(unhandled)
            self.json_file = "datablock.json"

        n_images = self._count_images()
        fast_mode = n_images < 10
        if fast_mode:
            info("%d images found, skipping a lot of processing" % n_images)

        self._find_spots()
        if not self._index():
            info("\nRetrying for stronger spots only...")
            os.rename("strong.pickle", "all_spots.pickle")
            self._find_spots(["sigma_strong=15"])
            if not self._index():
                warn("Giving up.")
                info(
                    """
Could not find an indexing solution. You may want to have a look
at the reciprocal space by running:

  dials.reciprocal_lattice_viewer datablock.json all_spots.pickle

or, to only include stronger spots:

  dials.reciprocal_lattice_viewer datablock.json strong.pickle
"""
                )
                sys.exit(1)

        if not fast_mode and not self._create_profile_model():
            info(
                "\nRefining model to attempt to increase number of valid "
                "spots..."
            )
            self._refine()
            if not self._create_profile_model():
                warn("Giving up.")
                info(
                    """
The identified indexing solution may not be correct. You may want to have a
look at the reciprocal space by running:

  dials.reciprocal_lattice_viewer experiments.json indexed.pickle
"""
                )
                sys.exit(1)

        if not fast_mode:
            self._check_intensities()
            self._report()

        if self.params.lower_bound_estimate.data == "integrated":
            self._integrate()
        self._wilson_calculation()

        self._refine_bravais()

        i19screen_runtime = timeit.default_timer() - start
        debug(
            "Finished at %s, total runtime: %.1f"
            % (time.strftime("%Y-%m-%d %H:%M:%S"), i19screen_runtime)
        )
        info(
            "i19.screen successfully completed (%.1f sec)" % i19screen_runtime
        )


if __name__ == "__main__":
    I19Screen().run()
