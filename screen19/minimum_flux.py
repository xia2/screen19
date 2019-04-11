#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

The Wilson plot of I as a function of d is drawn as the file
'wilson_plot.png'.  The plot can optionally be saved in other formats.

Examples:
  TODO add examples
"""

from __future__ import absolute_import, division, print_function

import sys
import logging
from tabulate import tabulate

import boost.python
import iotbx.phil
from dials.array_family import flex
import numpy as np
from scipy.optimize import curve_fit
from cctbx import miller
from dials.util.options import OptionParser
from screen19 import terminal_size, plot_intensities, d_ticks


# Suppress unhelpful matplotlib crash due to boost.python's overzealous allergy to FPEs
boost.python.floating_point_exceptions.division_by_zero_trapped = False

help_message = __doc__

phil_scope = iotbx.phil.parse(
    """
minimum_flux
  .caption = 'Parameters for the calculation of the lower flux bound'
  {
  desired_d = None
    .multiple = True
    .type = float
    .caption = u'Desired resolution limit, in Ångströms, of diffraction data'
    .help = 'This is the resolution target for the lower-bound flux ' \
            'recommendation.'
  min_i_over_sigma = 2
    .type = float
    .caption = u'Target I/σ value for lower-bound flux recommendation'
    .help = u'The lower-bound flux recommendation provides an estimate of ' \
            u'the flux required to ensure that the majority of expected ' \
            u'reflections at the desired resolution limit have I/σ greater ' \
            u'than or equal to this value.'
  wilson_fit_max_d = 4  # Å
    .type = float
    .caption = u'Maximum d-value (in Ångströms) for displacement parameter fit'
    .help = 'Reflections with lower resolution than this value will be ' \
            'ignored for the purposes of the Wilson plot.'
  }
output
  .caption = 'Parameters to control the output'
  {
  log = 'screen19.minimum_flux.log'
    .type = str
    .caption = 'Location for the info log'
  debug_log = 'screen19.minimum_flux.debug.log'
    .type = str
    .caption = 'Location for the debug log'
  wilson_plot = 'wilson_plot'
    .type = str
    .caption = 'Filename for the Wilson plot image'
    .help = "By default, the extension '.png' is appended.  If you include " \
            "a different extension, either '.pdf', '.ps', '.eps' or '.svg', " \
            "a file of that format will be created instead."
  }
verbosity = 1
  .type = int(value_min=0)
  .caption = "The verbosity level"
""",
    process_includes=True,
)

logger_name = "dials.screen19.minimum_flux"
logger = logging.getLogger(logger_name)
debug, info, warn = logger.debug, logger.info, logger.warning


def scaled_debye_waller(x, b, a):
    """
    TODO: Docstring

    :param x:
    :param b:
    :param a:
    :return:
    """
    return a * np.exp(-b / 2 * x)


def wilson_fit(d_star_sq, intensity, sigma, wilson_fit_max_d):
    """
    Fit a simple Debye-Waller factor, assume isotropic disorder parameter

    :param d_star_sq:
    :param intensity:
    :param sigma:
    :param wilson_fit_max_d:
    :return:
    """
    # Eliminate reflections with d > wilson_fit_max_d from the fit
    sel = d_star_sq > 1 / wilson_fit_max_d ** 2

    # Perform a weighted Wilson plot fit to the reflection intensities
    fit, cov = curve_fit(
        scaled_debye_waller,
        d_star_sq.select(sel),
        intensity.select(sel),
        sigma=sigma.select(sel),
        bounds=(0, np.inf),
    )

    return fit


def wilson_plot_ascii(crystal_symmetry, indices, intensity, sigma, d_ticks=None):
    """
    TODO: Docstring

    :param crystal_symmetry:
    :param indices:
    :param intensity:
    :param sigma:
    :param d_ticks:
    :return:
    """
    # Draw the Wilson plot, using existing functionality in cctbx.miller
    columns, rows = terminal_size()
    n_bins = min(columns, intensity.size())
    ms = miller.set(
        crystal_symmetry=crystal_symmetry, anomalous_flag=False, indices=indices
    )
    ma = miller.array(ms, data=intensity, sigmas=sigma)
    ma.set_observation_type_xray_intensity()
    ma.setup_binner_counting_sorted(n_bins=n_bins)
    wilson = ma.wilson_plot(use_binning=True)
    # Get the relevant plot data from the miller_array:
    binned_intensity = [x if x else 0 for x in wilson.data[1:-1]]
    bins = dict(zip(wilson.binner.bin_centers(1), binned_intensity))
    if d_ticks:
        tick_positions = ", ".join(['"%g" %s' % (d, 1 / d ** 2) for d in d_ticks])
        tick_positions = tick_positions.join(["(", ")"])
    else:
        tick_positions = ""
    # Draw the plot:
    plot_intensities(
        bins,
        1,
        title="'Wilson plot'",
        xlabel="'d (Angstrom) (inverse-square scale)'",
        ylabel="'I (counts)'",
        xticks=tick_positions,
        style="with lines",
    )


def wilson_plot_image(
    d_star_sq, intensity, fit, max_d=None, ticks=None, output="wilson_plot"
):
    """
    Generate the Wilson plot as an image, default is .png

    :param d_star_sq:
    :param intensity:
    :param fit:
    :param ticks:
    :param output:
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.xlabel(u"d (Å) (inverse-square scale)")
    plt.ylabel(u"Intensity (counts)")
    if ticks:
        plt.xticks([1 / d ** 2 for d in ticks], ["%g" % d for d in ticks])
    plt.yscale("log", nonposy="clip")
    plt.plot(d_star_sq, intensity, "b.", label=None)
    plt.plot(
        d_star_sq, scaled_debye_waller(d_star_sq, *fit), "r-", label="Debye-Waller fit"
    )
    if max_d:
        plt.fill_betweenx(
            plt.ylim(),
            1 / np.square(max_d),
            color="k",
            alpha=0.5,
            zorder=2.1,
            label="Excluded from fit",
        )
    plt.legend(loc=0)
    plt.savefig(output)
    plt.close()


def run(phil=phil_scope, args=None, set_up_logging=False):
    """
    TODO: Docstring

    :param phil:
    :param args:
    :return:
    """
    usage = "%prog [options] experiments.json reflections.pickle"

    parser = OptionParser(
        usage=usage,
        phil=phil,
        read_experiments=True,
        read_reflections=True,
        check_format=False,
        epilog=help_message,
    )

    params, options = parser.parse_args(args=args)

    if set_up_logging:
        from dials.util import log

        # Configure the logging
        log.config(
            params.verbosity, info=params.output.log, debug=params.output.debug_log
        )

    if not (params.input.experiments and params.input.reflections):
        print(help_message)
        sys.exit(1)

    if len(params.input.experiments) > 1:
        warn(
            "You provided more than one experiment list (%s).  Only the "
            "first, %s, will be used.",
            ", ".join([expt.filename for expt in params.input.experiments]),
            params.input.experiments[0].filename,
        )
    if len(params.input.reflections) > 1:
        warn(
            "You provided more than one reflection table (%s).  Only the "
            "first, %s, will be used.",
            ", ".join([refls.filename for refls in params.input.reflections]),
            params.input.reflections[0].filename,
        )

    expts = params.input.experiments[0].data
    refls = params.input.reflections[0].data
    # Ignore reflections without an index, since uctbx.unit_cell.d returns spurious
    # d == -1 values, rather than None, for unindexed reflections.
    refls.del_selected(refls["id"] == -1)
    # Ignore all spots flagged as overloaded
    refls.del_selected(refls.get_flags(refls.flags.overloaded).iselection())
    # The Wilson plot fit implicitly involves taking a logarithm of
    # intensities, so eliminate values that are going to cause problems
    try:
        # Work from profile-fitted intensities where possible
        refls = refls.select(refls["intensity.prf.value"] > 0)
    except RuntimeError:
        refls = refls.select(refls["intensity.sum.value"] > 0)

    if len(expts) > 1:
        warn(
            "The experiment list you provided, %s, contains more than one "
            "experiment object (perhaps multiple indexing solutions).  Only "
            "the first will be used, all others will be ignored.",
            params.input.experiments[0].filename,
        )

    # Parameters for the lower-bound flux estimate:
    min_i_over_sigma = params.minimum_flux.min_i_over_sigma
    wilson_fit_max_d = params.minimum_flux.wilson_fit_max_d
    desired_d = params.minimum_flux.desired_d
    # If no target resolution is given, use the following defaults:
    if not params.minimum_flux.desired_d:
        desired_d = [
            1,  # Å
            0.84,  # Å (IUCr publication requirement)
            0.6,  # Å
            0.4,  # Å
        ]
    desired_d.sort(reverse=True)

    # Get d-spacings, intensity & std dev of reflections
    symmetry = expts[0].crystal.get_crystal_symmetry()
    d_star_sq = 1 / symmetry.unit_cell().d(refls["miller_index"]) ** 2
    try:
        # Work from profile-fitted intensities and uncertainties where possible
        intensity = refls["intensity.prf.value"]
        sigma = flex.sqrt(refls["intensity.prf.variance"])
    except RuntimeError:
        intensity = refls["intensity.sum.value"]
        sigma = flex.sqrt(refls["intensity.sum.variance"])

    # Perform the Wilson plot fit
    fit = wilson_fit(d_star_sq, intensity, sigma, wilson_fit_max_d)

    # Get recommended dose factors
    # Use the fact that σ² = I for indexed data, so I/σ = √̅I
    desired_d_star_sq = [1 / d ** 2 for d in desired_d]
    recommended_factor = [
        (min_i_over_sigma ** 2 / scaled_debye_waller(target, *fit))
        for target in desired_d_star_sq
    ]

    # Draw the ASCII art Wilson plot
    wilson_plot_ascii(symmetry, refls["miller_index"], intensity, sigma, d_ticks)

    recommendations = zip(desired_d, recommended_factor)

    # Print a recommendation to the user.
    info("\nFitted isotropic displacement parameter, B = %.3g Angstrom^2", fit[0])
    for target, recommendation in recommendations:
        if recommendation <= 1:
            debug(
                "\nIt is likely that you can achieve a resolution of %g "
                "Angstrom using a lower flux.",
                target,
            )
        else:
            debug(
                "\nIt is likely that you need a higher flux to achieve a "
                "resolution of %g Angstrom.",
                target,
            )
        debug(
            "The estimated minimal sufficient flux is %.3g times the "
            "flux used for this data collection.",
            recommendation,
        )

    # TODO: SUggest what resolution is possible?

    summary = "\nRecommendations, summarised:\n"
    summary += tabulate(
        recommendations,
        ["Resolution\n(Angstrom)", "Suggested\ndose factor"],
        floatfmt=".3g",
        tablefmt="rst",
    )
    info(summary)

    # Draw the Wilson plot image and save to file
    wilson_plot_image(
        d_star_sq,
        intensity,
        fit,
        max_d=params.minimum_flux.wilson_fit_max_d,
        ticks=d_ticks,
        output=params.output.wilson_plot,
    )

    sys.exit(0)


if __name__ == "__main__":
    run(set_up_logging=True)
