# coding: utf-8

u"""
Process screening data obtained at Diamond Light Source Beamline I19.

This program presents the user with recommendations for adjustments to beam
flux, based on a single-sweep screening data collection.  It presents an
upper- and lower-bound estimate of suitable flux.
  • The upper-bound estimate is based on a comparison of a histogram of
  measured pixel intensities with the trusted intensity range of the detector.
  The user is warned when the measured pixel intensities indicate that the
  detector would have a significant number of overloaded or untrustworthy
  pixels.
  • The lower-bound estimate is based on a linear fit of isotropic disorder
  parameter, B, to a Wilson plot of reflection intensities.  From this,
  an estimate is made of the minimum exposure (flux × exposure time) required
  to achieve a target I/σ ratio (by default, target I/σ = 2) at one or more values
  of desired resolution, d, (by default, desired d = 1 Å, 0.84 Å, 0.6 Å & 0.4 Å).

Target I/σ and target d (in Ångström) can be set using the parameters
'min_i_over_sigma' and 'desired_d'.  One can set multiple values of the latter.

By default the disorder parameter fit is conducted on the
integrated data.  This ought to provide a reasonably true fit, but requires
an integration step, which can take some time.  You can achieve a quicker,
dirtier answer by fitting to the indexed data (i.e. only the stronger
spots), using 'minimum_exposure.data=indexed'.

Examples:

  screen19 imported_experiments.json

  screen19 *.cbf

  screen19 /path/to/data/

  screen19 /path/to/data/image0001.cbf:1:100

  screen19 min_i_over_sigma=2 desired_d=0.84 <imported_experiments.json | image_files>

  screen19 minimum_exposure.data=indexed <image_files>

"""

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import re
import sys
import time
import timeit
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import procrunner
from six.moves.cPickle import PickleError

import iotbx.phil
from libtbx import Auto
from libtbx.introspection import number_of_processors
from libtbx.phil import scope

import dials.command_line.integrate
import dials.util.version
from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.bravais_settings import (
    refined_settings_from_refined_triclinic,
)
from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex
from dials.command_line.dials_import import MetaDataUpdater
from dials.command_line.index import index
from dials.command_line.refine import run_dials_refine
from dials.command_line.refine_bravais_settings import (
    bravais_lattice_to_space_group_table,
    eliminate_sys_absent,
    map_to_primitive,
)
from dials.util import Sorry, log, version
from dials.util.ascii_art import spot_counts_per_image_plot
from dials.util.options import OptionParser
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import (
    BeamComparison,
    DetectorComparison,
    ExperimentListFactory,
    ExperimentListTemplateImporter,
    GoniometerComparison,
)

import screen19
from screen19.minimum_exposure import suggest_minimum_exposure

Templates = List[Tuple[str, Tuple[int, int]]]

phil_scope = iotbx.phil.parse(
    u"""
    verbosity = 0
        .type = int(value_min=0)
        .caption = 'Verbosity level of log output'
        .help = "Possible values:\n"
                "\t• 0: Info log output to stdout/logfile\n"
                "\t• 1: Info & debug log output to stdout/logfile"

    output
        .caption = 'Options to control the output files'
        {
        log = 'screen19.log'
        .type = str
        .caption = "The log filename"
        }
    nproc = Auto
        .type = int
        .caption = 'Number of processors to use'
        .help = "The chosen value will apply to all the DIALS utilities with a "
                "multi-processing option.  If 'False' or 'Auto', all available "
                "processors will be used."

    minimum_exposure
        .caption = 'Options for screen19.minimum_exposure'
        {
        include scope screen19.minimum_exposure.phil_scope
        data = indexed *integrated
            .type = choice
            .caption = 'Choice of data for the displacement parameter fit'
            .help = 'For the lower-bound exposure estimate, choose whether to use '
                    'indexed (quicker) or integrated (better) data in fitting '
                    'the isotropic displacement parameter.'
        }

    maximum_flux
        .caption = 'Options for avoiding detector paralysation'
        {
        trusted_range_correction = 0.25
            .type = float(value_min=0, value_max=1)
            .caption = 'Factor by which to multiply the maximum trusted flux.'
            .help = "The detector manufacturer's photon count correction, to correct "
                    "for pixel paralysation, is often found to be unreliable at photon "
                    "counts in the upper part of the nominal trusted range.  In such "
                    "cases, this factor can be used to adjust the upper limit of the "
                    "trusted range.  Pilatus detectors, for example, have been found "
                    "not to give reliable correction for photon counts greater than "
                    "0.25 × the manufacturer's trusted range.  It is therefore "
                    "sensible to present the user with a correspondingly reduced upper-"
                    "limit flux recommendation."
        }

    dials_import
        .caption = 'Options for dials.import'
        {
        include scope dials.command_line.dials_import.phil_scope

        input
            {
            include scope dials.util.options.tolerance_phil_scope

            experiments = None
                .help = "The experiment list file path"
                .type = str
                .multiple = True
                .optional = True
            }
        }

    dials_find_spots
        .caption = 'Options for dials.find_spots'
        {
        include scope dials.command_line.find_spots.phil_scope
        }

    dials_index
        .caption = 'Options for dials.index'
        {
        include scope dials.command_line.index.phil_scope
        }

    dials_refine
        .caption = 'Options for dials.refine'
        {
        include scope dials.command_line.refine.phil_scope
        }

    dials_refine_bravais
        .caption = 'Options for dials.refine_bravais_settings'
        {
        include scope dials.command_line.refine_bravais_settings.phil_scope
        }

    dials_create_profile
        .caption = 'Options for dials.create_profile_model'
        {
        include scope dials.command_line.create_profile_model.phil_scope
        }

    dials_integrate
        .caption = 'Options for dials.integrate'
        {
        include scope dials.command_line.integrate.phil_scope
        }

    dials_report
        .caption = 'Options for dials.report'
        {
        include scope dials.command_line.report.phil_scope
        }
    """,
    process_includes=True,
)

procrunner_debug = False

logger = logging.getLogger("dials.screen19")
debug, info, warning = logger.debug, logger.info, logger.warning


def _run_integration(scope, experiments_file, reflections_file):
    # type: (scope, str, str) -> Tuple[ExperimentList, flex.reflection_table]
    """Run integration programatically, compatible with multiple DIALS versions.

    Args:
        scope: The dials.integrate phil scope
        experiments_file: Path to the experiment list file
        reflections_file: Path to the reflection table file
    """

    if hasattr(dials.command_line.integrate, "run_integration"):
        # DIALS 3.1+ interface
        expts, refls, _ = dials.command_line.integrate.run_integration(
            scope.extract(),
            ExperimentList.from_file(experiments_file),
            flex.reflection_table.from_file(reflections_file),
        )
    elif hasattr(dials.command_line.integrate, "Script"):
        # Pre-3.1-style programmatic interface
        expts, refls = dials.command_line.integrate.Script(phil=scope).run(
            [experiments_file, reflections_file]
        )
    else:
        raise RuntimeError(
            "Could not find dials.integrate programmatic interface 'run_integration' or 'Script'"
        )

    return expts, refls


def overloads_histogram(d_spacings, ticks=None, output="overloads"):
    # type: (Sequence[float], Optional[Sequence[float]], Optional[str]) -> None
    """
    Generate a histogram of reflection d-spacings as an image, default is .png.

    Args:
        d_spacings:  d-spacings of the reflections.
        ticks (optional):  d-values for the tick positions on the 1/d axis.
        output (optional):  Output filename root, to which the extension `.png` will
                            be appended.  Default is `overloads`.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.xlabel(u"d (Å) (inverse scale)")
    plt.ylabel(u"Number of overloaded reflections")
    if ticks:
        plt.xticks([1 / d for d in ticks], ["%g" % d for d in ticks])

    # Matplotlib v3.3.0 includes API change 'nonposy' → 'nonpositive'
    # https://matplotlib.org/api/api_changes.html#log-symlog-scale-base-ticks-and-nonpos-specification
    try:
        plt.yscale("log", nonpositive="clip")
    except ValueError:
        plt.yscale("log", nonposy="clip")

    plt.hist(d_spacings, min(100, d_spacings.size()))
    plt.savefig(output)
    plt.close()


class Screen19(object):
    """Encapsulates the screening script."""

    def __init__(self):
        # Throughout the pipeline, retain the state of the processing.
        self.expts = ExperimentList([])
        self.refls = flex.reflection_table()
        # Get some default parameters.  These must be extracted from the 'fetched'
        # PHIL scope, rather than the 'definition' phil scope returned by
        # iotbx.phil.parse.  Confused?  Blame PHIL.
        self.params = phil_scope.fetch(iotbx.phil.parse("")).extract()

    def _quick_import(self, files):  # type: (List[str]) -> bool
        """
        Generate xia2-style templates from file names and attempt a quick import.

        From each given filename, generate a filename template by substituting a hash
        character (#) for each numeral in the last contiguous group of numerals
        before the file extension.  For example, the filename `example_01_0001.cbf`
        becomes `example_01_####.cbf`.

        Contiguous image ranges are recorded by associating the start and end image
        number of the range with the relevant filename template.

        dials.import is then run with options to extrapolate header information from
        the first image file, thereby running more quickly than reading each image
        header individually.

        Args:
            files:  List of image filenames.

        Returns:
            Boolean flag indicating whether the quick import has succeeded.
        """
        if len(files) == 1:
            # No point in quick-importing a single file
            return False
        debug("Attempting quick import...")
        files.sort()
        templates = {}  # type: Dict[str, List[Optional[List[int]]]]
        for f in files:
            template, image = screen19.make_template(f)
            if template not in templates:
                image_range = [image, image] if image else []
                templates.update({template: [image_range]})
            elif image == templates[template][-1][-1] + 1:
                templates[template][-1][-1] = image
            elif image == templates[template][-1][-1]:
                # We have a duplicate input file name.  Do nothing.
                pass
            else:
                templates[template].append([image, image])
        # Return tuple of template and image range for each unique image range
        templates = [
            (t, tuple(r)) for t, ranges in templates.items() for r in ranges
        ]  # type: Templates
        return self._quick_import_templates(templates)

    def _quick_import_templates(self, templates):  # type: (Templates) -> bool
        """
        Take image file templates and frame number ranges and try to run dials.import.

        dials.import is run with options to extrapolate header information from
        the first image file, thereby running more quickly than reading each image
        header individually.

        Args:
            templates:  A list of tuples, each tuple containing a xia2-style filename
                        template and the start and end image numbers of the associated
                        sweep.

        Returns:
            Boolean flag indicating whether the quick import has succeeded.
        """
        debug("Quick import template summary:\n\t%s", templates)
        if len(templates) > 1:
            debug("Cannot currently run quick import on multiple templates.")
            return False

        try:
            scan_range = templates[0][1]  # type: Tuple[int, int]
            if not scan_range:
                raise IndexError
        except IndexError:
            debug("Cannot run quick import: could not determine image naming template.")
            return False

        info("Running quick import.")
        self.params.dials_import.input.template = [templates[0][0]]
        self.params.dials_import.geometry.scan.image_range = scan_range
        self.params.dials_import.geometry.scan.extrapolate_scan = True
        self._run_dials_import()

        return True

    def _import(self, files):  # type: (List[str]) -> None
        """
        Try to run a quick call of dials.import.  Failing that, run a slow call.

        Try initially to construct file name templates contiguous groups of files.
        Failing that, pass a full list of the files to the importer (slower).

        Args:
            files:  List of image filenames.
        """
        info("\nImporting data...")
        if len(files) == 1:
            if os.path.isdir(files[0]):
                debug(
                    "You specified a directory. Importing all CBF files in "
                    "that directory."
                )
                # TODO Support HDF5.
                files = [
                    os.path.join(files[0], f)
                    for f in os.listdir(files[0])
                    if f.endswith(".cbf")
                    or f.endswith(".cbf.gz")
                    or f.endswith(".cbf.bz2")
                ]
            elif len(files[0].split(":")) == 3:
                debug(
                    "You specified an image range in the xia2 format.  "
                    "Importing all specified files."
                )
                template, start, end = files[0].split(":")
                template = screen19.make_template(template)[0]
                start, end = int(start), int(end)
                if not self._quick_import_templates([(template, (start, end))]):
                    warning("Could not import specified image range.")
                    sys.exit(1)
                info("Quick import successful.")
                return
            elif files[0].endswith(".expt"):
                debug(
                    "You specified an existing experiment list file.  "
                    "No import necessary."
                )
                try:
                    self.expts = ExperimentList.from_file(files[0])
                except (IOError, PickleError, ValueError):
                    pass
                else:
                    self.params.dials_import.output.experiments = files[0]
                    if self.expts:
                        return

        if not files:
            warning("No images found matching input.")
            sys.exit(1)

        # Can the files be quick-imported?
        if self._quick_import(files):
            info("Quick import successful.")
            return

        self.params.dials_import.input.experiments = files
        self._run_dials_import()

    def _run_dials_import(self):
        """
        Perform a minimal version of dials.import to get an experiment list.

        Use some filleted bits of dials.import and dials.util.options.Importer.
        """
        # Get some key data format arguments.
        try:
            format_kwargs = {
                "dynamic_shadowing": self.params.dials_import.format.dynamic_shadowing,
                "multi_panel": self.params.dials_import.format.multi_panel,
            }
        except AttributeError:
            format_kwargs = {}

        # If filenames contain wildcards, expand
        args = []
        for arg in self.params.dials_import.input.experiments:
            if "*" in arg:
                args.extend(glob(arg))
            else:
                args.append(arg)

        if args:
            # Are compare{beam,detector,goniometer} and scan_tolerance necessary?
            # They are cargo-culted from the DIALS option parser.
            tol_params = self.params.dials_import.input.tolerance
            compare_beam = BeamComparison(
                wavelength_tolerance=tol_params.beam.wavelength,
                direction_tolerance=tol_params.beam.direction,
                polarization_normal_tolerance=tol_params.beam.polarization_normal,
                polarization_fraction_tolerance=tol_params.beam.polarization_fraction,
            )
            compare_detector = DetectorComparison(
                fast_axis_tolerance=tol_params.detector.fast_axis,
                slow_axis_tolerance=tol_params.detector.slow_axis,
                origin_tolerance=tol_params.detector.origin,
            )
            compare_goniometer = GoniometerComparison(
                rotation_axis_tolerance=tol_params.goniometer.rotation_axis,
                fixed_rotation_tolerance=tol_params.goniometer.fixed_rotation,
                setting_rotation_tolerance=tol_params.goniometer.setting_rotation,
            )
            scan_tolerance = tol_params.scan.oscillation

            # Import an experiment list from image data.
            try:
                experiments = ExperimentListFactory.from_filenames(
                    args,
                    compare_beam=compare_beam,
                    compare_detector=compare_detector,
                    compare_goniometer=compare_goniometer,
                    scan_tolerance=scan_tolerance,
                    format_kwargs=format_kwargs,
                )
            except IOError as e:
                warning("%s '%s'", e.strerror, e.filename)
                sys.exit(1)

            # Record the imported experiments for use elsewhere.
            # Quit if there aren't any.
            self.expts.extend(experiments)
            if not self.expts:
                warning("No images found.")
                sys.exit(1)

        else:
            # Use the template importer.
            if len(self.params.dials_import.input.template) > 0:
                importer = ExperimentListTemplateImporter(
                    self.params.dials_import.input.template, format_kwargs=format_kwargs
                )
                # Record the imported experiments for use elsewhere.
                # Quit if there aren't any.
                self.expts.extend(importer.experiments)
                if not self.expts:
                    warning(
                        "No images found matching template %s"
                        % self.params.dials_import.input.template[0]
                    )
                    sys.exit(1)

        # Setup the metadata updater
        metadata_updater = MetaDataUpdater(self.params.dials_import)

        # Extract the experiments and loop through
        self.expts = metadata_updater(self.expts.imagesets())

    def _count_processors(self, nproc=None):  # type: (Optional[int]) -> None
        """
        Determine the number of processors and save it as an instance variable.

        The user may specify the number of processors to use.  If no value is
        given, the number of available processors is returned.

        Args:
            nproc (optional):  Number of processors.
        """
        if nproc and nproc is not Auto:
            self.nproc = nproc
            return

        # if environmental variable NSLOTS is set to a number then use that
        try:
            self.nproc = int(os.environ.get("NSLOTS"))
            return
        except (ValueError, TypeError):
            pass

        self.nproc = number_of_processors(return_value_if_unknown=-1)

        if self.nproc <= 0:
            warning(
                "Could not determine number of available processors. Error code %d",
                self.nproc,
            )
            sys.exit(1)

    def _count_images(self):  # type: () -> int
        """
        Attempt to determine the number of diffraction images.

        The number of diffraction images is determined from the imported_experiments
        JSON file.

        Returns:
            Number of images.
        """
        # FIXME:  This exception handling should be redundant.  Empty experiment
        #         lists should get caught at the import stage.  Is this so?
        try:
            return self.expts[0].imageset.size()
        except IndexError:
            warning("Could not determine number of images in dataset.")
            sys.exit(1)

    def _check_intensities(self, mosaicity_correction=True):  # type: (bool) -> None
        """
        Run xia2.overload and plot a histogram of pixel intensities.

        If `mosaicity_correction` is true, the pixel intensities are approximately
        adjusted to take account of a systematic defect in the detector count rate
        correction.  See https://github.com/xia2/screen19/wiki#mosaicity-correction

        Args:
            mosaicity_correction (optional):  default is `True`.
        """
        info("\nTesting pixel intensities...")
        command = ["xia2.overload", "nproc=%s" % self.nproc, "indexed.expt"]
        debug("running %s", command)
        result = procrunner.run(command, print_stdout=False, debug=procrunner_debug)
        debug("result = %s", screen19.prettyprint_procrunner(result))
        info("Successfully completed (%.1f sec)", result["runtime"])

        if result["exitcode"] != 0:
            warning("Failed with exit code %d", result["exitcode"])
            sys.exit(1)

        with open("overload.json") as fh:
            overload_data = json.load(fh)

        info("Pixel intensity distribution:")
        count_sum = 0
        hist = {}
        if "bins" in overload_data:
            for b in range(overload_data["bin_count"]):
                if overload_data["bins"][b] > 0:
                    hist[b] = overload_data["bins"][b]
                    count_sum += b * overload_data["bins"][b]
        else:
            hist = {int(k): v for k, v in overload_data["counts"].items() if int(k) > 0}
            count_sum = sum([k * v for k, v in hist.items()])

        average_to_peak = 1
        if mosaicity_correction:
            # Adjust for the detector count rate correction
            if self._sigma_m:
                delta_z = self._oscillation / self._sigma_m / math.sqrt(2)
                average_to_peak = (
                    math.sqrt(math.pi) * delta_z * math.erf(delta_z)
                    + math.exp(-(delta_z ** 2))
                    - 1
                ) / delta_z ** 2
                info("Average-to-peak intensity ratio: %f", average_to_peak)

        scale = 100 * overload_data["scale_factor"] / average_to_peak
        info("Determined scale factor for intensities as %f", scale)

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
                rescaled_hist[rescaled] = hist[x] + rescaled_hist.get(rescaled, 0)
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

        screen19.plot_intensities(
            hist, 1 / hist_granularity, procrunner_debug=procrunner_debug
        )

        linear_response_limit = 100 * self.params.maximum_flux.trusted_range_correction
        marginal_limit = max(70, linear_response_limit)

        text = "".join(
            (
                "Strongest pixel (%d counts) " % max_count,
                "reaches %.1f%% " % hist_max,
                "of the detector count rate limit",
            )
        )
        if hist_max > 100:
            warning("Warning: %s!", text)
        else:
            info(text)
        if (
            "overload_limit" in overload_data
            and max_count >= overload_data["overload_limit"]
        ):
            warning(
                "Warning: THE DATA CONTAIN REGULAR OVERLOADS!\n"
                "         The photon incidence rate is outside the specified "
                "limits of the detector.\n"
                "         The built-in detector count rate correction cannot "
                "adjust for this.\n"
                "         You should aim for count rates below {:.0%} of the "
                "detector limit.".format(
                    self.params.maximum_flux.trusted_range_correction
                )
            )
        elif hist_max > marginal_limit:
            warning(
                "Warning: The photon incidence rate is well outside the "
                "linear response region of the detector (<{:.0%}).\n"
                "    The built-in detector count rate correction may not be "
                "able to adjust for this.".format(
                    self.params.maximum_flux.trusted_range_correction
                )
            )
        elif hist_max > linear_response_limit:
            info(
                "The photon incidence rate is outside the linear response "
                "region of the detector (<{:.0%}).\n"
                "    The built-in detector count rate correction may be able "
                "to adjust for this.".format(
                    self.params.maximum_flux.trusted_range_correction
                )
            )
        if not mosaicity_correction:
            warning(
                "Warning: Not enough data for proper profile estimation."
                "    The spot intensities are not corrected for mosaicity.\n"
                "    The true photon incidence rate will be higher than the "
                "given estimate."
            )

        info("Total sum of counts in dataset: %d", count_sum)

    def _find_spots(self, args=None):  # type: (Optional[List[str]]) -> None
        """
        Call `dials.find_spots` on the imported experiment list.

        Args:
            args (optional):  List of any additional PHIL parameters to be used by
                              dials.import.
        """
        info("\nFinding spots...")

        dials_start = timeit.default_timer()

        # Use some choice fillets from dials.find_spots
        # Ignore `args`, use `self.params`

        # Loop through all the imagesets and find the strong spots

        self.refls = flex.reflection_table.from_observations(
            self.expts, self.params.dials_find_spots
        )

        # Add n_signal column - before deleting shoeboxes

        good = MaskCode.Foreground | MaskCode.Valid
        self.refls["n_signal"] = self.refls["shoebox"].count_mask_values(good)

        # Delete the shoeboxes
        if not self.params.dials_find_spots.output.shoeboxes:
            del self.refls["shoebox"]

        info(
            60 * "-" + "\n%s\n" + 60 * "-" + "\nSuccessfully completed (%.1f sec)",
            spot_counts_per_image_plot(self.refls),
            timeit.default_timer() - dials_start,
        )

    def _index(self):  # type: () -> bool
        """
        Call `dials.index` on the output of spot finding.

        Returns:
            Boolean value indicating whether indexing was successful.
        """
        dials_start = timeit.default_timer()

        # Prepare max_cell constraint strategies.
        max_cell = self.params.dials_index.indexing.max_cell
        # By default, try unconstrained max_cell followed by max_cell=20.
        # If the user has already specified a max_cell < 20, do not relax to 20Å.
        cell_constraints = [([], max_cell)]
        if not max_cell or max_cell is Auto or max_cell > 20:
            cell_constraints += [(["max_cell constraint"], 20)]

        # Prepare indexing methods, preferring the real_space_grid_search if a
        # known unit cell has been specified, otherwise using 3D FFT, then 1D FFT.
        methods = (
            [(["real space grid search"], "real_space_grid_search")]
            if self.params.dials_index.indexing.known_symmetry.unit_cell
            else []
        )
        methods += [(["3D FFT"], "fft3d"), (["1D FFT"], "fft1d")]

        # Cycle through the indexing methods for each of the max_cell constraint
        # strategies until an indexing solution is found.
        for i, (max_cell_msg, max_cell) in enumerate(cell_constraints):
            # Set the max_cell constraint strategy.
            self.params.dials_index.indexing.max_cell = max_cell
            for j, (method_msg, method) in enumerate(methods):
                # Set the indexing method.
                self.params.dials_index.indexing.method = method
                # Log a handy message to the user.
                msg = (
                    "Retrying with " + " and ".join(method_msg + max_cell_msg)
                    if i + j
                    else "Indexing"
                )
                info("\n%s...", msg)
                try:
                    # If indexing is successful, break out of the inner loop.
                    self.expts, self.refls = index(
                        self.expts, [self.refls], self.params.dials_index
                    )
                    break
                except (DialsIndexError, ValueError) as e:
                    # If indexing is unsuccessful, try again with the next
                    # strategy.
                    warning("Failed: %s", str(e))
                    continue
            else:
                # When all the indexing methods are unsuccessful, move onto
                # the next max_cell constraint strategy and try again.
                continue
            # We should only get here if successfully indexed. Break out of the loop
            break
        else:
            # Indexing completely unsuccessful.
            return False

        sg_type = self.expts[0].crystal.get_crystal_symmetry().space_group().type()
        symb = sg_type.universal_hermann_mauguin_symbol()
        unit_cell = self.expts[0].crystal.get_unit_cell()

        self.refls.as_file(self.params.dials_index.output.reflections)
        self.expts.as_file(self.params.dials_index.output.experiments)
        self.refls.as_file(self.params.dials_index.output.reflections)
        info(
            "Found primitive solution: %s %s using %s reflections\n"
            "Indexed experiments and reflections saved as %s, %s\n"
            "Successfully completed (%.1f sec)",
            symb,
            unit_cell,
            self.refls["id"].count(0),
            self.params.dials_index.output.experiments,
            self.params.dials_index.output.reflections,
            timeit.default_timer() - dials_start,
        )

        # Report the indexing successful.
        return True

    def _wilson_calculation(self):  # type: () -> None
        """
        Run `screen19.minimum_exposure` on an experiment list and reflection table.

        For best results, the reflections and experiment list should contain the
        results of integration or scaling.  If only strong spots are used, the Wilson
        plot fit may be poor.
        """
        dials_start = timeit.default_timer()
        info("\nEstimating lower exposure bound...")

        suggest_minimum_exposure(self.expts, self.refls, self.params.minimum_exposure)

        info("Successfully completed (%.1f sec)", timeit.default_timer() - dials_start)

    def _refine(self):  # type: () -> None
        """
        Run `dials.refine` on the results of indexing.
        """
        dials_start = timeit.default_timer()
        info("\nRefining...")

        try:
            self.expts, self.refls, _, _ = run_dials_refine(
                self.expts, self.refls, self.params.dials_refine
            )
        except Sorry as e:
            warning("dials.refine failed: %d\nGiving up.\n", e)
            sys.exit(1)

        info("Successfully refined (%.1f sec)", timeit.default_timer() - dials_start)

    def _create_profile_model(self):  # type: () -> bool
        """
        Run `dials.create_profile_model` on indexed reflections.

        The indexed experiment list will be overwritten with a copy that includes
        the profile model but is otherwise identical.

        Returns:
            Boolean value indicating whether it was possible to determine a profile
            model from the data.
        """
        info("\nCreating profile model...")
        command = [
            "dials.create_profile_model",
            self.params.dials_index.output.experiments,
            self.params.dials_index.output.reflections,
            "output = %s" % self.params.dials_index.output.experiments,
        ]
        result = procrunner.run(command, print_stdout=False, debug=procrunner_debug)
        debug("result = %s", screen19.prettyprint_procrunner(result))
        self._sigma_m = None
        if result["exitcode"] == 0:
            db = ExperimentList.from_file(self.params.dials_index.output.experiments)[0]
            self._oscillation = db.imageset.get_scan().get_oscillation()[1]
            self._sigma_m = db.profile.sigma_m()
            info(
                u"%d images, %s° oscillation, σ_m=%.3f°",
                db.imageset.get_scan().get_num_images(),
                str(self._oscillation),
                self._sigma_m,
            )
            info("Successfully completed (%.1f sec)", result["runtime"])
            return True
        warning("Failed with exit code %d", result["exitcode"])
        return False

    def _integrate(self):  # type: () -> None
        """Run `dials.integrate` to integrate reflection intensities."""
        dials_start = timeit.default_timer()
        info("\nIntegrating...")

        # Don't waste time recreating the profile model
        self.params.dials_integrate.create_profile_model = False
        # Get the dials.integrate PHIL scope, populated with parsed input parameters
        integrate_scope = phil_scope.get("dials_integrate").objects[0]
        integrate_scope.name = ""
        integrate_scope = integrate_scope.format(self.params.dials_integrate)

        try:
            integrated_experiments, integrated_reflections = _run_integration(
                integrate_scope,
                self.params.dials_index.output.experiments,
                self.params.dials_index.output.reflections,
            )
            # Save the output to files
            integrated_reflections.as_file(
                self.params.dials_integrate.output.reflections
            )
            integrated_experiments.as_file(
                self.params.dials_integrate.output.experiments
            )
            # ... and also store the output internally
            self.expts, self.refls = integrated_experiments, integrated_reflections
            info(
                "Successfully completed (%.1f sec)",
                timeit.default_timer() - dials_start,
            )
        except SystemExit as e:
            if e.code:
                warning("dials.integrate failed with exit code %d\nGiving up.", e.code)
                sys.exit(1)

    # This is a hacky check but should work for as long as DIALS 2.0 is supported.
    if version.dials_version() < "DIALS 2.1":

        def _refine_bravais(self, experiments, reflections):
            # type: (ExperimentList, flex.reflection_table) -> None
            """
            Run `dials.refine_bravais_settings` on an experiments and reflections.

            Args:
                experiments:  An experiment list..
                reflections:  The corresponding reflection table.
            """
            info("\nRefining Bravais settings...")
            command = ["dials.refine_bravais_settings", experiments, reflections]
            result = procrunner.run(command, print_stdout=False, debug=procrunner_debug)
            debug("result = %s", screen19.prettyprint_procrunner(result))
            if result["exitcode"] == 0:
                m = re.search(
                    r"[-+]{3,}\n[^\n]*\n[-+|]{3,}\n(.*\n)*[-+]{3,}",
                    result["stdout"].decode("utf-8"),
                )
                if m:
                    info(m.group(0))
                else:
                    info(
                        "Could not interpret dials.refine_bravais_settings output, "
                        "please check dials.refine_bravais_settings.log"
                    )
                info("Successfully completed (%.1f sec)", result["runtime"])
            else:
                warning("Failed with exit code %d", result["exitcode"])
                sys.exit(1)

    else:

        def _refine_bravais(self):  # type: () -> None
            """Run `dials.refine_bravais_settings` to determine the space group."""
            dials_start = timeit.default_timer()
            info("\nRefining Bravais settings...")

            self.refls = eliminate_sys_absent(self.expts, self.refls)
            map_to_primitive(self.expts, self.refls)

            try:
                refined_settings = refined_settings_from_refined_triclinic(
                    self.expts, self.refls, self.params.dials_refine_bravais
                )
            except RuntimeError as e:
                warning("dials.refine_bravais_settings failed.\nGiving up.")
                sys.exit(e)

            possible_bravais_settings = {
                solution["bravais"] for solution in refined_settings
            }
            bravais_lattice_to_space_group_table(possible_bravais_settings)
            try:
                # Old version of dials with as_str() method
                logger.info(refined_settings.as_str())
            except AttributeError:
                # Newer versions of dials (>= 2.2.2) has proper __str__ method
                logger.info(refined_settings)

            info(
                "Successfully completed (%.1f sec)",
                timeit.default_timer() - dials_start,
            )

    def _report(self, experiments, reflections):
        # type: (ExperimentList, flex.reflection_table) -> None
        """
        Run `dials.report` on an experiment list and reflection table.

        Args:
            experiments:  An experiment list.
            reflections:  The corresponding reflection table.
        """
        info("\nCreating report...")
        command = ["dials.report", experiments, reflections]
        result = procrunner.run(command, print_stdout=False, debug=procrunner_debug)
        debug("result = %s", screen19.prettyprint_procrunner(result))
        if result["exitcode"] == 0:
            info("Successfully completed (%.1f sec)", result["runtime"])
        #     if sys.stdout.isatty():
        #       info("Trying to start browser")
        #       try:
        #         import subprocess
        #         d = dict(os.environ)
        #         d["LD_LIBRARY_PATH"] = ""
        #         subprocess.Popen(["xdg-open", "dials-report.html"], env=d)
        #       except Exception as e:
        #         debug("Could not open browser\n%s", str(e))
        else:
            warning("Failed with exit code %d", result["exitcode"])
            sys.exit(1)

    def run(self, args=None, phil=phil_scope, set_up_logging=False):
        # type: (Optional[List[str]], scope, bool) -> None
        """
        TODO: Docstring.

        Args:
            args:
            phil:
            set_up_logging:

        Returns:

        """
        usage = "%prog [options] image_directory | image_files.cbf | imported.expt"

        parser = OptionParser(
            usage=usage, epilog=__doc__, phil=phil, check_format=False
        )

        self.params, options, unhandled = parser.parse_args(
            args=args, show_diff_phil=True, return_unhandled=True, quick_parse=True
        )

        version_information = "screen19 v%s using %s (%s)" % (
            screen19.__version__,
            dials.util.version.dials_version(),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        start = timeit.default_timer()

        if len(unhandled) == 0:
            print(__doc__)
            print(version_information)
            return

        if set_up_logging:
            # Configure the logging
            log.config(verbosity=self.params.verbosity, logfile=self.params.output.log)
            # Unless verbose output has been requested, suppress generation of
            # debug and info log records from any child DIALS command, retaining
            # those from screen19 itself.
            if not self.params.verbosity:
                logging.getLogger("dials").setLevel(logging.WARNING)
                logging.getLogger("dials.screen19").setLevel(logging.INFO)

        info(version_information)
        debug("Run with:\n%s\n%s", " ".join(unhandled), parser.diff_phil.as_str())

        self._count_processors(nproc=self.params.nproc)
        debug("Using %s processors", self.nproc)
        # Set multiprocessing settings for spot-finding, indexing and
        # integration to match the top-level specified number of processors
        self.params.dials_find_spots.spotfinder.mp.nproc = self.nproc
        self.params.dials_index.indexing.nproc = self.nproc
        # Setting self.params.dials_refine.refinement.mp.nproc is not helpful
        self.params.dials_integrate.integration.mp.nproc = self.nproc

        # Set the input and output parameters for the DIALS components
        # TODO: Compare to diff_phil and start from later in the pipeline if
        #  appropriate
        self._import(unhandled)
        imported_name = self.params.dials_import.output.experiments

        self._find_spots()

        if not self._index():
            info("\nRetrying for stronger spots only...")
            strong_refls = self.refls
            self.params.dials_find_spots.spotfinder.threshold.dispersion.sigma_strong = (
                15
            )
            self._find_spots()

            if not self._index():
                warning("Giving up.")
                self.expts.as_file(imported_name)
                strong_refls.as_file("strong.refl")
                self.refls.as_file("stronger.refl")
                info(
                    "Could not find an indexing solution. You may want to "
                    "have a look at the reciprocal space by running:\n\n"
                    "    dials.reciprocal_lattice_viewer %s %s\n\n"
                    "or, to only include stronger spots:\n\n"
                    "    dials.reciprocal_lattice_viewer %s %s\n",
                    imported_name,
                    "strong.refl",
                    imported_name,
                    "stronger.refl",
                )
                sys.exit(1)

        if not self._create_profile_model():
            info("\nRefining model to attempt to increase number of valid spots...")
            self._refine()
            if not self._create_profile_model():
                warning("Giving up.")
                info(
                    "The identified indexing solution may not be correct. "
                    "You may want to have a look at the reciprocal space by "
                    "running:\n\n"
                    "    dials.reciprocal_lattice_viewer indexed.expt indexed.refl\n"
                )
                sys.exit(1)

        self._check_intensities()

        if self.params.minimum_exposure.data == "integrated":
            self._integrate()

            self._wilson_calculation()

            experiments = self.params.dials_integrate.output.experiments
            reflections = self.params.dials_integrate.output.reflections
        else:
            self._wilson_calculation()

            experiments = self.params.dials_create_profile.output
            reflections = self.params.dials_index.output.reflections

        # This is a hacky check but should work for as long as DIALS 2.0 is supported.
        if version.dials_version() < "DIALS 2.1":
            self._refine_bravais(experiments, reflections)
        else:
            self._refine_bravais()

        self._report(experiments, reflections)

        runtime = timeit.default_timer() - start
        debug(
            "Finished at %s, total runtime: %.1f",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            runtime,
        )
        info("screen19 successfully completed (%.1f sec).", runtime)


def main():  # type: () -> None
    """Dispatcher for command-line call."""
    Screen19().run(set_up_logging=True)
