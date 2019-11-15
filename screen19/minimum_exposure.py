# coding: utf-8
u"""
Perform straight-line Wilson plot fit.  Draw the Wilson plot.

Reflection d-spacings are determined from the crystal symmetry (from
indexing) and the Miller indices of the indexed reflections.  The
atomic displacement parameter is assumed isotropic.  Its value is
determined from a fit to the reflection data:
  I = A * exp(-B / (2 * d^2)),
where I is the intensity and the scale factor, A, and isotropic
displacement parameter, B, are the fitted parameters.

An I/σ condition for 'good' diffraction statistics is set by the
instance variable min_i_over_sigma, and the user's desired
resolution is set by the instance variable desired_d.  A crude
error model is assumed, whereby σ² = I, and so the I/σ condition
translates trivially to a threshold I.

The value of the fitted intensity function at the desired
resolution is compared with the threshold I.  The ratio of these
values is used to determine a recommended exposure (flux × exposure time)
for the full data collection.

The Wilson plot of I as a function of d is drawn as the file
'wilson_plot.png'.  The plot can optionally be saved in other formats.

Examples:

    screen19.minimum_exposure integrated_experiments.json integrated.pickle

    screen19.minimum_exposure indexed_experiments.json indexed.pickle

    screen19.minimum_exposure min_i_over_sigma=2 desired_d=0.84 wilson_fit_max_d=4 \
        integrated_experiments.json integrated.pickle

"""

from __future__ import absolute_import, division, print_function

import sys
import logging
import numpy as np
from tabulate import tabulate
from typing import Iterable, List, Optional, Sequence, Union  # noqa: F401
# Flake8 does not detect typing yet (https://gitlab.com/pycqa/flake8/issues/342)

import boost.python
from cctbx import crystal, miller
from dials.array_family import flex
from dials.util.options import OptionParser
from dxtbx.model import Experiment, ExperimentList
import iotbx.phil
from libtbx.phil import scope, scope_extract
from scipy.optimize import curve_fit
from screen19 import d_ticks, dials_v1, plot_intensities, terminal_size


# Custom types
FloatSequence = Sequence[float]
Fit = Union[np.ndarray, Iterable, int, float]

# Suppress unhelpful matplotlib crash due to boost.python's overzealous allergy to FPEs
boost.python.floating_point_exceptions.division_by_zero_trapped = False

help_message = __doc__


if dials_v1:
    verbosity_scope = u"""
    verbosity = 1
        .type = int(value_min=0)
        .caption = 'The verbosity level of the command-line output'
        .help = "Possible values:\n"
                "\t• 0: Suppress all command-line output;\n"
                "\t• 1: Show regular output on the command line;\n"
                "\t• 2: Show regular output, plus detailed debugging messages."
    """
else:
    verbosity_scope = u"""
    verbosity = 0
        .type = int(value_min=0)
        .caption = 'Verbosity level of log output'
        .help = "Possible values:\n"
                "\t• 0: Info log output to stdout/logfile\n"
                "\t• 1: Info & debug log output to stdout/logfile"
    """

phil_scope = iotbx.phil.parse(
    u"""
    minimum_exposure
        .caption = 'Parameters for the calculation of the lower exposure bound'
        {
        desired_d = None
            .multiple = True
            .type = float
            .caption = u'Desired resolution limit, in Ångströms, of diffraction data'
            .help = 'This is the resolution target for the lower-bound exposure ' \
                    'recommendation.'
        min_i_over_sigma = 2
            .type = float
            .caption = u'Target I/σ value for lower-bound exposure recommendation'
            .help = 'The lower-bound exposure recommendation provides an estimate of ' \
                    u'the exposure (flux × exposure time) required to ensure that the' \
                    'majority of expected reflections at the desired resolution limit' \
                    u'have I/σ greater than or equal to this value.'
        wilson_fit_max_d = 4  # Å
            .type = float
            .caption = u'Maximum d-value (in Ångströms) for displacement parameter fit'
            .help = 'Reflections with lower resolution than this value will be ' \
                    'ignored for the purposes of the Wilson plot.'
        }
    output
        .caption = 'Parameters to control the output'
        {
        log = 'screen19.minimum_exposure.log'
            .type = str
            .caption = 'Location for the info log'
        debug_log = 'screen19.minimum_exposure.debug.log'
            .type = str
            .caption = 'Location for the debug log'
        wilson_plot = 'wilson_plot'
            .type = str
            .caption = 'Filename for the Wilson plot image'
            .help = "By default, the extension '.png' is appended.  If you include " \
                    "a different extension, either '.pdf', '.ps', '.eps' or '.svg', " \
                    "a file of that format will be created instead."
        }
        """ + verbosity_scope,
    process_includes=True,
)

logger_name = "dials.screen19.minimum_exposure"
logger = logging.getLogger(logger_name)
debug, info, warn = logger.debug, logger.info, logger.warning


def scaled_debye_waller(x, b, a):
    # type: (float, float, float) -> float
    u"""
    Calculate a scaled isotropic Debye-Waller factor.

    By assuming a single isotropic disorder parameter, :param:`b`, this factor
    approximates the decay of diffracted X-ray intensity increasing resolution
    (decreasing d, increasing sin(θ)).

    Args:
        x: Equivalent to 1/d².
        b: Isotropic displacement parameter.
        a: A scale factor.

    Returns:
        Estimated value of scaled isotropic Debye-Waller factor.
    """
    return a * np.exp(-b / 2 * x)


def wilson_fit(d_star_sq, intensity, sigma, wilson_fit_max_d):
    # type: (FloatSequence, FloatSequence, FloatSequence, float) -> Fit
    u"""
    Fit a simple Debye-Waller factor, assume isotropic disorder parameter.

    Reflections with d ≥ :param:`wilson_fit_max_d` are ignored.

    Args:
        d_star_sq: 1/d² (equivalently d*²), sequence of values for the observed
            reflections (units of Å⁻² assumed).
        intensity: Sequence of reflection intensities.
        sigma: Sequence of uncertainties in reflection intensity.
        wilson_fit_max_d: The minimum resolution for reflections against which to
            fit.

    Returns:
        - The fitted isotropic displacement parameter (units of Å² assumed);
        - The fitted scale factor.

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


def wilson_plot_ascii(
        crystal_symmetry,  # type: crystal.symmetry
        indices,  # type: Sequence[flex.miller_index, ...]
        intensity,  # type: FloatSequence
        sigma,  # type: FloatSequence
        d_ticks=None,  # type: Optional[FloatSequence]
):
    # type: (...) -> None
    u"""
    Print an ASCII-art Wilson plot of reflection intensities.

    Equivalent reflections will be merged according to the crystal symmetry.

    Args:
        crystal_symmetry: Crystal symmetry.
        indices: Miller indices of reflections.
        intensity: Intensities of reflections.
        sigma: Standard uncertainties in reflection intensities.
        d_ticks: d location of ticks on 1/d² axis.
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
    d_star_sq,  # type: FloatSequence
    intensity,  # type: FloatSequence
    fit,  # type: Fit
    max_d=None,  # type: Optional[float]
    ticks=None,  # type: Optional[FloatSequence]
    output="wilson_plot"  # type: str
):
    # type: (...) -> None
    u"""
    Generate the Wilson plot as a PNG image.

    :param:`max_d` allows greying out of the reflections not included in the
    isotropic Debye-Waller fit.

    Args:
        d_star_sq: 1/d² values of reflections.
        intensity: Intensities of reflections.
        fit: Fitted parameters (tuple of fitted isotropic displacement parameter and
            fitted scale factor).
        max_d: The minimum resolution for reflections used in the Debye-Waller fit.
        ticks: d location of ticks on 1/d² axis.
        output: Output filename.  The extension `.png` will be added automatically.
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


def suggest_minimum_exposure(expts, refls, params):
    # type: (ExperimentList[Experiment], flex.reflection_table, scope_extract) -> None
    u"""
    Suggest an estimated minimum sufficient exposure to achieve a certain resolution.

    The estimate is based on a fit of a Debye-Waller factor under the assumption that a
    single isotropic displacement parameter can be used to adequately describe the
    decay of intensities with increasing sin(θ).

    An ASCII-art Wilson plot is printed, along with minimum exposure recommendations for
    a number of different resolution targets.  The Wilson plot, including the fitted
    isotropic Debye-Waller factor, is saved as a PNG image.

    Args:
        expts: Experiment list containing a single experiment, from which the crystal
            symmetry will be extracted.
        refls: Reflection table of observed reflections.
        params: Parameters for calculation of minimum exposure estimate.
    """
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

    # Parameters for the lower-bound exposure estimate:
    min_i_over_sigma = params.minimum_exposure.min_i_over_sigma
    wilson_fit_max_d = params.minimum_exposure.wilson_fit_max_d
    desired_d = params.minimum_exposure.desired_d
    # If no target resolution is given, use the following defaults:
    if not params.minimum_exposure.desired_d:
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

    # Get recommended exposure factors
    # Use the fact that σ² = I for indexed data, so I/σ = √̅I
    desired_d_star_sq = [1 / d ** 2 for d in desired_d]
    target_i = min_i_over_sigma ** 2
    recommended_factor = [
        (target_i / scaled_debye_waller(target_d, *fit))
        for target_d in desired_d_star_sq
    ]

    # Get the achievable resolution at the current exposure
    desired_d += [np.sqrt(fit[0] / (2 * np.log(fit[1] / target_i)))]
    recommended_factor += [1]

    # Draw the ASCII art Wilson plot
    wilson_plot_ascii(symmetry, refls["miller_index"], intensity, sigma, d_ticks)

    recommendations = zip(desired_d, recommended_factor)
    recommendations = sorted(recommendations, key=lambda rec: rec[0], reverse=True)

    # Print a recommendation to the user.
    info(u"\nFitted isotropic displacement parameter, B = %.3g Å²", fit[0])
    for target, recommendation in recommendations:
        if recommendation < 1:
            debug(
                u"\nIt is likely that you can achieve a resolution of %g Å using a "
                "lower exposure (lower transmission and/or shorter exposure time).",
                target,
            )
        elif recommendation > 1:
            debug(
                "\nIt is likely that you need a higher exposure (higher transmission "
                u"and/or longer exposure time to achieve a resolution of %g Å.",
                target,
            )
        debug(
            u"The estimated minimal sufficient exposure (flux × exposure time) to "
            u"achievea resolution of %.2g Å is %.3g times the exposure used for this "
            "data collection.",
            target, recommendation,
        )

    summary = "\nRecommendations, summarised:\n"
    summary += tabulate(
        recommendations,
        [u"Resolution (Å)", "Suggested\nexposure factor"],
        floatfmt=(".2g", ".3g"),
        tablefmt="rst",
    )
    summary += u"\nExposure is flux × exposure time." \
               "\nYou can achieve your desired exposure factor by modifying " \
               "transmission and/or exposure time."
    info(summary)

    # Draw the Wilson plot image and save to file
    wilson_plot_image(
        d_star_sq,
        intensity,
        fit,
        max_d=params.minimum_exposure.wilson_fit_max_d,
        ticks=d_ticks,
        output=params.output.wilson_plot,
    )


def run(phil=phil_scope, args=None, set_up_logging=False):
    # type: (scope, Optional[List[str, ...]], bool) -> None
    """
    Parse command-line arguments, run the script.

    Uses the DIALS option parser to extract an experiment list, reflection table and
    parameters, then passes them to :func:`suggest_minimum_exposure`.
    Optionally, sets up the logger.

    Args:
        phil: PHIL scope for option parser.
        args: Arguments to parse. If None, :data:`sys.argv[1:]` will be used.
        set_up_logging: Choose whether to configure :module:`screen19` logging.
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

    if len(expts) > 1:
        warn(
            "The experiment list you provided, %s, contains more than one "
            "experiment object (perhaps multiple indexing solutions).  Only "
            "the first will be used, all others will be ignored.",
            params.input.experiments[0].filename,
        )

    suggest_minimum_exposure(expts, refls, params)


def main():
    # type: () -> None
    """Dispatcher for command-line call."""
    run(set_up_logging=True)
