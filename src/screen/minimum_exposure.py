"""
Perform straight-line Wilson plot fit.  Draw the Wilson plot.
"""

from __future__ import annotations

help_message = """
Reflection d-spacings are determined from the crystal symmetry (from
indexing) and the Miller indices of the indexed reflections.  The
atomic displacement parameter is assumed isotropic.  Its value is
determined from a fit to the reflection data: \n
  I = A * exp(-B / (2 * d^2)), \n
where I is the intensity and the scale factor, A, and isotropic
displacement parameter, B, are the fitted parameters. \n
\n
An I/σ condition for 'good' diffraction statistics is set by the
instance variable min_i_over_sigma, and the user's desired
resolution is set by the instance variable desired_d.  A crude
error model is assumed, whereby σ² = I, and so the I/σ condition
translates trivially to a threshold I. \n
\n
The value of the fitted intensity function at the desired
resolution is compared with the threshold I.  The ratio of these
values is used to determine a recommended exposure (flux × exposure time)
for the full data collection. \n
\n
The Wilson plot of I as a function of d is drawn as the file
'wilson_plot.png'.  The plot can optionally be saved in other formats.
\n
Examples:\n

    screen19.minimum_exposure integrated.expt integrated.refl \n

    screen19.minimum_exposure indexed.expt indexed.refl \n

    screen19.minimum_exposure min_i_over_sigma=2 desired_d=0.84 wilson_fit_max_d=4 \
        integrated.expt integrated.refl \n

"""

import argparse
import logging
import sys
from io import StringIO
from typing import Iterable, Sequence, Union

import numpy as np
from scipy.optimize import curve_fit
from tabulate import tabulate

import libtbx.phil
from cctbx import miller

from dials.array_family import flex
from dials.util import log
from dxtbx.model import ExperimentList

from screen import (  # FIXME TODO change to relative import
    config_parser,
    d_ticks,
    plot_intensities,
    terminal_size,
    version_parser,
)

# Custom types
FloatSequence = Sequence[float]
Fit = Union[np.ndarray, Iterable, int, float]
Scope = libtbx.phil.scope
ScopeExtract = libtbx.phil.scope_extract


phil_scope = libtbx.phil.parse(
    """
    minimum_exposure
      .caption = 'Parameters for the calculation of the lower exposure bound'
      {
      data = indexed *integrated
        .type = choice
        .caption = ''
        .help = 'Choice of data for the displacement parameter fit'
          'For the lower-bound exposure estimate, choose whether to use '
          'indexed (quicker) or integrated (better) data in fitting '
          'the isotropic displacement parameter.'
      desired_d = None
        .multiple = True
        .type = float
        .help = u'Desired resolution limit, in Ångströms, of diffraction data' \
          'This is the resolution target for the lower-bound exposure recommendation.'
      min_i_over_sigma = 2
        .type = float
        .help = u'Target I/σ value for lower-bound exposure recommendation' \
          'The lower-bound exposure recommendation provides an estimate of ' \
          u'the exposure (flux × exposure time) required to ensure that the' \
          'majority of expected reflections at the desired resolution limit' \
          u'have I/σ greater than or equal to this value.'
      wilson_fit_max_d = 4  # Å
        .type = float
        .help = u'Maximum d-value (in Ångströms) for displacement parameter fit' \
          'Reflections with lower resolution than this value will be ' \
          'ignored for the purposes of the Wilson plot.'
    }
    output {
      verbosity = 0
        .type = int(value_min=0)
        .help = "Verbosity level of log output. Possible values:\n"
          "\t• 0: Info log output to stdout/logfile\n"
          "\t• 1: Info log output to stdout/logfile, logfile contains timing"
          "information\n"
          "\t• 2: Info & debug log output to stdout/logfile, logfile contains"
          "timing information"
      log = 'screen19.minimum_exposure.log'
        .type = str
        .help = 'Location for the info log'
      wilson_plot = 'wilson_plot'
        .type = str
        .help = "Filename for the Wilson plot image. By default, the extension '.png' is appended. " \
          "If you include a different extension, either '.pdf', '.ps', '.eps' or '.svg', " \
          "a file of that format will be created instead."
    }
    """,
    process_includes=True,
)

logger_name = "dials.screen19.minimum_exposure"
logger = logging.getLogger(logger_name)
debug, info, warn = logger.debug, logger.info, logger.warning


def scaled_debye_waller(x: float, b: float, a: float) -> float:
    """
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


def wilson_fit(
    d_star_sq: FloatSequence,
    intensity: FloatSequence,
    sigma: FloatSequence,
    wilson_fit_max_d: float,
) -> Fit:
    """
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
    sel = d_star_sq > 1 / wilson_fit_max_d**2

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
    miller_array: miller.array, d_ticks: Sequence | None = None
) -> None:
    """
    Print an ASCII-art Wilson plot of reflection intensities.

    Equivalent reflections will be merged according to the crystal symmetry.

    Args:
        miller_array: An array of integrated intensities, bundled with appropriate
                      crystal symmetry and unit cell info.
        d_ticks: d location of ticks on 1/d² axis.
    """
    # Draw the Wilson plot, using existing functionality in cctbx.miller
    columns, rows = terminal_size()
    n_bins = min(columns, miller_array.data().size())
    miller_array.setup_binner_counting_sorted(n_bins=n_bins, reflections_per_bin=1)
    wilson = miller_array.wilson_plot(use_binning=True)
    # Get the relevant plot data from the miller_array:
    binned_intensity = [x if x else 0 for x in wilson.data[1:-1]]
    bins = dict(zip(wilson.binner.bin_centers(1), binned_intensity))
    if d_ticks:
        tick_positions = ", ".join([f'"{d:g}" {1 / d ** 2}' for d in d_ticks])
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
    d_star_sq: FloatSequence,
    intensity: FloatSequence,
    fit: Fit,
    max_d: float | None = None,
    ticks: FloatSequence | None = None,
    output: str = "wilson_plot",
) -> None:
    """
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

    plt.xlabel("d (Å) (inverse-square scale)")
    plt.ylabel("Intensity (counts)")
    if ticks:
        plt.xticks([1 / d**2 for d in ticks], [f"{d:g}" for d in ticks])

    # Matplotlib v3.3.0 includes API change 'nonposy' → 'nonpositive'
    # https://matplotlib.org/api/api_changes.html#log-symlog-scale-base-ticks-and-nonpos-specification
    try:
        plt.yscale("log", nonpositive="clip")
    except ValueError:
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


def suggest_minimum_exposure(
    expts: ExperimentList, refls: flex.reflection_table, params: ScopeExtract
) -> None:
    """
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

    if params.minimum_exposure.data == "integrated":
        # Work from profile-fitted intensities where possible but if the number of
        # profile-fitted intensities is less than 75% of the number of summed
        # intensities, use summed intensities instead.  This is a very arbitrary heuristic.
        sel_prf = refls.get_flags(refls.flags.integrated_prf).iselection()
        sel_sum = refls.get_flags(refls.flags.integrated_sum).iselection()
        if sel_prf.size() < 0.75 * sel_sum.size():
            refls = refls.select(sel_sum)
            intensity = refls["intensity.sum.value"]
            sigma = flex.sqrt(refls["intensity.sum.variance"])
        else:
            refls = refls.select(sel_prf)
            intensity = refls["intensity.prf.value"]
            sigma = flex.sqrt(refls["intensity.prf.variance"])
    else:
        # This is still a bit rough
        intensity = refls["intensity.sum.value"]
        sigma = flex.sqrt(refls["intensity.sum.variance"])

    # Apply French-Wilson scaling to ensure positive intensities.
    miller_array = miller.array(
        miller.set(
            expts.crystal.get_crystal_symmetry(),
            refls["miller_index"],
            anomalous_flag=False,
        ),
        data=intensity,
        sigmas=sigma,
    )
    miller_array.set_observation_type_xray_intensity()
    miller_array = miller_array.merge_equivalents().array()
    cctbx_log = StringIO()  # Prevent idiosyncratic CCTBX logging from polluting stdout.
    miller_array = miller_array.french_wilson(log=cctbx_log).as_intensity_array()
    logger.debug(cctbx_log.getvalue())

    d_star_sq = miller_array.d_star_sq().data()
    intensity = miller_array.data()
    sigma = miller_array.sigmas()

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

    # Perform the Wilson plot fit
    fit = wilson_fit(d_star_sq, intensity, sigma, wilson_fit_max_d)

    # Get recommended exposure factors
    # Use the fact that σ² = I for indexed data, so I/σ = √̅I
    desired_d_star_sq = [1 / d**2 for d in desired_d]
    target_i = min_i_over_sigma**2
    recommended_factor = [
        (target_i / scaled_debye_waller(target_d, *fit))
        for target_d in desired_d_star_sq
    ]

    # Get the achievable resolution at the current exposure
    desired_d += [np.sqrt(fit[0] / (2 * np.log(fit[1] / target_i)))]
    recommended_factor += [1]

    # Draw the ASCII art Wilson plot
    wilson_plot_ascii(miller_array, d_ticks)

    recommendations = zip(desired_d, recommended_factor)
    recommendations = sorted(recommendations, key=lambda rec: rec[0], reverse=True)

    # Print a recommendation to the user.
    info("\nFitted isotropic displacement parameter, B = %.3g Å²", fit[0])
    for target, recommendation in recommendations:
        if recommendation < 1:
            debug(
                "\nIt is likely that you can achieve a resolution of %g Å using a "
                "lower exposure (lower transmission and/or shorter exposure time).",
                target,
            )
        elif recommendation > 1:
            debug(
                "\nIt is likely that you need a higher exposure (higher transmission "
                "and/or longer exposure time to achieve a resolution of %g Å.",
                target,
            )
        debug(
            "The estimated minimal sufficient exposure (flux × exposure time) to "
            "achievea resolution of %.2g Å is %.3g times the exposure used for this "
            "data collection.",
            target,
            recommendation,
        )

    summary = "\nRecommendations summarised:\n"
    summary += tabulate(
        recommendations,
        ["Resolution (Å)", "Suggested\nexposure factor"],
        floatfmt=(".3g", ".3g"),
        tablefmt="rst",
    )
    summary += (
        "\nExposure is flux × exposure time."
        "\nYou can achieve your desired exposure factor by modifying "
        "transmission and/or exposure time."
    )
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


usage = "%(prog)s [options] integrated.expt integrated.refl"
parser = argparse.ArgumentParser(
    usage=usage,
    formatter_class=argparse.RawTextHelpFormatter,
    description=__doc__,
    epilog=help_message,
    parents=[version_parser, config_parser],
)
parser.add_argument("expt", type=str, help="Experiment file")
parser.add_argument("refl", type=str, help="Reflection file")
parser.add_argument("phil_args", nargs="*", help="Phil parameters for fit.")


def run(set_up_logging: bool = False):
    """
    Parse command-line arguments, run the script. Optionally, sets up the logger.

    Args:
        set_up_logging (bool, optional): Choose whether to configure `screen19` logging. Defaults to False.
    """
    args = parser.parse_args()
    cl = phil_scope.command_line_argument_interpreter()
    working_phil = phil_scope.fetch(cl.process_and_fetch(args.phil_args))

    if args.show_config:
        # FIXME doesn't work unless some words are passed as positional argument (experiments)
        working_phil.show(attributes_level=args.attributes_level)
        sys.exit()

    params = working_phil.extract()

    if set_up_logging:
        # Configure the logging
        log.config(params.output.verbosity, params.output.log)

    expt = ExperimentList.from_file(args.expt)
    refl = flex.reflection_table.from_file(args.refl)

    if len(expt) > 1:
        warn(
            f"The experiment list you provided, {args.expt}, contains more than one "
            "experiment object (perhaps multiple indexing solutions).  Only "
            "the first will be used, all others will be ignored."
        )

    suggest_minimum_exposure(expt[0], refl, params)


def main() -> None:
    """Dispatcher for command-line call."""
    run(set_up_logging=True)
