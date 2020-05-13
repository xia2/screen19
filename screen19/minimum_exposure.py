# coding: utf-8
u"""
Perform straight-line Wilson plot fit.  Draw the Wilson plot.

Reflection d-spacings are determined from the crystal symmetry (from
indexing) and the Miller indices of the indexed reflections.  The
atomic displacement parameter is assumed isotropic.  Its value is
determined from a fit to the reflection data:
  log(<I>/Σ) = log(A) - B / (2 * d²),
where <I> is the experimental intensity and Σ is the model intensity
averaged within resolution shells. The scale factor, A, and isotropic
displacement parameter, B, are the fitted parameters.

An I/σ condition for 'good' diffraction statistics is set by the
instance variable min_i_over_sigma, and the user's desired
resolution is set by the instance variable desired_d.

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

import logging
from math import exp
import numpy as np
from tabulate import tabulate
import time

# Flake8 does not detect typing yet (https://gitlab.com/pycqa/flake8/issues/342)
from typing import Iterable, List, Optional, Sequence, Union  # noqa: F401

import boost.python
import cctbx.eltbx.xray_scattering
from dials.array_family import flex
from dials.util.options import OptionParser
from dials.util.version import dials_version
from dxtbx.model import Experiment, ExperimentList
import iotbx.phil
from libtbx.phil import scope, scope_extract
from screen19 import d_ticks, dials_v1, plot_wilson, __version__


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
        dstarsq_bin_size = 0.004
            .type = float
            .caption = u'Bin size for averaging intensity data.'
            .help = 'Bin size in 1/d² units used for averaging intensity data' \
                    'in calculation of Wilson plot.'
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
        """
    + verbosity_scope,
    process_includes=True,
)

logger_name = "dials.screen19.minimum_exposure"
logger = logging.getLogger(logger_name)
debug, info, warn = logger.debug, logger.info, logger.warning


def number_residues_estimate(symmetry):
    u"""
    Guess the number of residues in the asymmetric unit cell, assuming most frequent
    Matthews coefficient 2.34 Å^3/Da at 50% solvent content, 112.5 Da average residue weight and
    average residue composition from http://www.ccp4.ac.uk/html/matthews_coef.html.
    """
    sg = symmetry.space_group()
    uc = symmetry.unit_cell()

    n_ops = len(sg.all_ops())

    v_asu = uc.volume() / n_ops
    n_res = int(round(v_asu / (2.34 * 112.5)))
    asu_contents = {"C": 5 * n_res,
                    "N": 1.35 * n_res,
                    "O": 1.5 * n_res,
                    "H": 8 * n_res,
                    "S": 0.05 * n_res
                    }
    scattering_factors = {}
    for atom in asu_contents.keys():
        scattering_factors[atom] = cctbx.eltbx.xray_scattering.wk1995(atom).fetch()

    return asu_contents, scattering_factors


def model_f_sq(stol_sq, asu_contents, scattering_factors, symmetry):
    u"""
    Compute expected model intensity values in resolution shells
    """
    sum_fj_sq = 0
    for atom, n_atoms in asu_contents.items():
        f0 = scattering_factors[atom].at_stol_sq(stol_sq)
        sum_fj_sq += f0 * f0 * n_atoms
    sum_fj_sq *= symmetry.space_group().order_z() \
                        * symmetry.space_group().n_ltr()
    return sum_fj_sq

def wilson_fit(iobs, asu_contents, scattering_factors, symmetry, wilson_fit_max_d):
    u"""
    Fit a simple Debye-Waller factor, assume isotropic disorder parameter.
    Adapted from cctbx implementation in cctbx.statistics.wilson_plot object.

    Reflections with d ≥ :param:`wilson_fit_max_d` are ignored.

    Args:
        iobs: Sequence of observed reflection intensities.
        asu_contents: Dictionary with atom quantities in asymmetric unit cell.
        scattering_factors: Dictionary with atomic scattering factors.
        symmetry: Crystal symmetry object.
        wilson_fit_max_d: The minimum resolution for reflections against which to
            fit.

    Returns:
        - The fitted isotropic displacement parameter (units of Å² assumed);
        - Logarithm of the fitted scale factor.
        - List of graph values for Wilson Plot.

    """
    assert iobs.is_real_array()
    # compute <fobs^2> in resolution shells
    mean_iobs = iobs.mean(
        use_binning=True,
        use_multiplicities=True).data[1:-1]
    n_none = mean_iobs.count(None)
    if (n_none > 0):
        error_message = "wilson_plot error: number of empty bins: %d" % n_none
        info = iobs.info()
        if (info is not None):
            error_message += "\n  Info: " + str(info)
        error_message += "\n  Number of bins: %d" % len(mean_iobs)
        raise RuntimeError(error_message)
    mean_iobs = flex.double(mean_iobs)
    # compute <s^2> = <(sin(theta)/lambda)^2> in resolution shells
    stol_sq = iobs.sin_theta_over_lambda_sq()
    stol_sq.use_binner_of(iobs)
    mean_stol_sq = flex.double(stol_sq.mean(
        use_binning=True,
        use_multiplicities=True).data[1:-1])
    # compute expected f_calc^2 in resolution shells
    icalc = flex.double()
    for stol_sq in mean_stol_sq:
        sum_fj_sq = model_f_sq(stol_sq, asu_contents, scattering_factors, symmetry)
        icalc.append(sum_fj_sq)
    # fit to straight line
    x = mean_stol_sq
    y = flex.log(mean_iobs / icalc)
    sigma = flex.sqrt(mean_iobs)
    idx_resol = [i for i, v in enumerate(x) if v < 1. / (4 * wilson_fit_max_d**2)][-1]
    try:
        idx_nan = next((i for i,(v,s) in enumerate(zip(list(y), list(sigma))) if (np.isnan(v) or
                                                                                  np.isnan(s) or
                                                                                  np.isinf(v) or
                                                                                  np.isinf(s))))
    except StopIteration:
        idx_nan = len(y)
    sel_x = list(x)[idx_resol:idx_nan]
    sel_y = list(y)[idx_resol:idx_nan]
    fit = flex.linear_regression(flex.double(sel_x), flex.double(sel_y))
    assert fit.is_well_defined()
    fit_y_intercept = fit.y_intercept()
    fit_slope = fit.slope()
    wilson_b = -fit_slope / 2
    return wilson_b, fit_y_intercept, x, y, (idx_resol, idx_nan)


def wilson_plot_ascii(
    stol_sq,  # type: Sequence[flex.miller_index, ...]
    intensity,  # type: FloatSequence
    wilson_b,  # type: float
    fit_y_intercept,  # type: float
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
    plot_data = [(s, v, fit_y_intercept - 2 * wilson_b * s) for s, v in zip(stol_sq, intensity)]
    if d_ticks:
        tick_positions = ", ".join(['"%g" %s' % (d, 1 / (4 * d ** 2)) for d in d_ticks])
        tick_positions = tick_positions.join(["(", ")"])
    else:
        tick_positions = ""
    # Draw the plot:
    plot_wilson(
        plot_data,
        1,
        title="'Wilson plot'",
        xlabel="'d (Angstrom) (inverse-square scale)'",
        ylabel="'<I>/Σ'",
        xticks=tick_positions,
    )


def wilson_plot_image(
    stol_sq,  # type: FloatSequence
    log_i_over_sig,  # type: FloatSequence
    wilson_b,  # type: float
    fit_y_intercept,  # type: FloatSequence
    idx_d_range=None,  # type: Optional[float]
    output="wilson_plot",  # type: str
):
    # type: (...) -> None
    u"""
    Generate the Wilson plot as a PNG image.

    :param:`max_d` allows greying out of the reflections not included in the
    isotropic Debye-Waller fit.

    Args:
        stol_sq: (sin(θ)/λ)² values of reflections.
        log_i_over_sig: Log of intensities of reflections log(<I>/Σ).
        wilson_b: Fitted isotropic displacement parameter.
        fit_y_intercept: Fitted scale factor.
        idx_d_range: indices corresponding to a resolution range used in the Debye-Waller fit.
        ticks: d location of ticks on (sin(θ)/λ)² axis.
        output: Output filename.  The extension `.png` will be added automatically.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    plt.scatter(stol_sq, log_i_over_sig, c='b', s=5)
    plt.plot(stol_sq,[-2 * wilson_b * v + fit_y_intercept for v in stol_sq])
    plt.xlabel(u"d / Å")
    plt.ylabel(u"log(<I>/Σ)")
    ax.set_xticklabels(
            ["{:.2f}".format(np.float64(1.0) / (2 * np.sqrt(x))) if x > 0 else np.inf for x in ax.get_xticks()]
    )
    max_stol_sq, min_stol_sq = tuple(stol_sq[i] for i in idx_d_range)
    max_d, min_d = tuple((1.0 / (2 * np.sqrt(x)) for x in (max_stol_sq, min_stol_sq)))
    plt.title(u"Fitted isotropic displacement parameter, B = %.3g Å²\n Wilson Plot fitting range: %.3g - %.3g Å" % (wilson_b, max_d, min_d))
    try:
        y_range = plt.ylim()
        plt.fill_betweenx(
            y_range,
            max_stol_sq,
            color="k",
            alpha=0.5,
            zorder=2.1,
            label="Excluded from fit",
        )
        plt.fill_betweenx(
            y_range,
            min_stol_sq,
            plt.xlim()[-1],
            color="k",
            alpha=0.5,
            zorder=2.1
        )
    except TypeError:
        pass
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
        iobs = refls.as_miller_array(expts[0], intensity="prf").resolution_filter(d_max=100, d_min=0)
    except:
        iobs = refls.as_miller_array(expts[0], intensity="sum").resolution_filter(d_max=100, d_min=0)
    iobs.setup_binner_d_star_sq_step(d_star_sq_step=params.minimum_exposure.dstarsq_bin_size)

    # Parameters for the lower-bound exposure estimate:
    min_i_over_sigma = params.minimum_exposure.min_i_over_sigma
    wilson_fit_max_d = params.minimum_exposure.wilson_fit_max_d
    desired_d = params.minimum_exposure.desired_d

    # Get estimated asymmetric unit cell contents and corresponding scattering factors
    symmetry = expts[0].crystal.get_crystal_symmetry()
    asu_contents, scattering_factors = number_residues_estimate(symmetry)

    # Perform the Wilson plot fit
    wilson_b, fit_y_intercept, x, y, idx_fit_range = wilson_fit(iobs,
                                                                asu_contents,
                                                                scattering_factors,
                                                                symmetry,
                                                                wilson_fit_max_d)
    max_d, min_d = tuple((1.0 / (2 * np.sqrt(x[i])) for i in idx_fit_range))
    
    # Get reference resolution from I/σ value
    iobs_selected = iobs.select(iobs.data() > 0)
    iobs_selected.use_binning_of(iobs) 
    mean_i_over_sigma = iobs_selected.i_over_sig_i(use_binning=True, return_fail=0)
    min_i_over_sigma_bin = next((res for val, res in zip(mean_i_over_sigma.data[1:-1],
                                                         mean_i_over_sigma.binner.range_all()[1:-1]) if val < min_i_over_sigma))
    dmin_i_over_sigma = mean_i_over_sigma.binner.bin_d_min(min_i_over_sigma_bin)
    
    # If no target resolution is given, use the following defaults:
    if not params.minimum_exposure.desired_d:
        desired_d = [dmin_i_over_sigma,]
        desired_d.extend((round(dmin_i_over_sigma * 10 / sc) / 10. for sc in (1.25, 1.5, 1.75, 2, dmin_i_over_sigma)))
    desired_d = sorted(set(desired_d), reverse=True)[:5]

    # Get recommended exposure factors
    recommended_factor = [
        exp(-wilson_b / (2.*dmin_i_over_sigma**2) + wilson_b / (2.*target_d**2)) \
        * model_f_sq(1. / (4.*dmin_i_over_sigma**2), asu_contents, scattering_factors, symmetry) \
        / model_f_sq(1. / (4.*target_d**2), asu_contents, scattering_factors, symmetry)
        for target_d in desired_d
    ]

    # Draw the ASCII art Wilson plot
    wilson_plot_ascii(x, y, wilson_b, fit_y_intercept, d_ticks)
    
    recommendations = zip(desired_d, recommended_factor)
    recommendations = sorted(recommendations, key=lambda rec: rec[0], reverse=True)

    # Print a recommendation to the user.
    info(u"\nFitted isotropic displacement parameter, B = %.3g Å²", wilson_b)
    info(u"\nWilson Plot fitting range: %.3g - %.3g Å", max_d, min_d)
    info(u"\nSelected reference resolution %.3g Å at I/σ = %.2g", dmin_i_over_sigma, min_i_over_sigma)
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
            u"achieve a resolution of %.2g Å is %.3g times the exposure used for this "
            "data collection.",
            target,
            recommendation,
        )
    
    summary = "\nRecommendations, summarised:\n"
    summary += tabulate(
        recommendations,
        [u"Resolution (Å)", "Suggested\nexposure factor"],
        floatfmt=(".2f", ".2g"),
        tablefmt="rst",
    )
    summary += (
        u"\nExposure is flux × exposure time."
        "\nYou can achieve your desired exposure factor by modifying "
        "transmission and/or exposure time."
    )
    info(summary)

    # Draw the Wilson plot image and save to file
    wilson_plot_image(
        x,
        y,
        wilson_b,
        fit_y_intercept,
        idx_fit_range,
        params.output.wilson_plot,
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
        if dials_v1:
            log.config(
                params.verbosity, info=params.output.log, debug=params.output.debug_log
            )
        else:
            log.config(params.verbosity, params.output.log)

    if not (params.input.experiments and params.input.reflections):
        version_information = "screen19.minimum_exposure v%s using %s (%s)" % (
            __version__,
            dials_version(),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        print(help_message)
        print(version_information)
        return

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
