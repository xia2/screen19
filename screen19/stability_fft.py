from __future__ import absolute_import, division, print_function

import time

import iotbx.phil
from scitbx.array_family import flex

help_message = """

Examples::

  i19.stability_fft image_*.cbf

"""

phil_scope = iotbx.phil.parse(
    """\
plot = False
  .type = bool
  .help = "Make some plots"
remove_spots = False
  .type = bool
  .help = "Remove signal pixels from images before analysis"
scan_range = None
  .help = "Scan range to use for analysis"
  .type = ints(size=2)
exposure_time = None
  .type = float
  .help = "Override exposure time if not correctly written in headers"
output_file = "i19_stability.dat"
  .type = path
  .help = "Name for the output file"

""",
    process_includes=True,
)


def stability_fft(imageset, params):
    from i19.util.time_analysis import fft

    scan = imageset.get_scan()
    detector = imageset.get_detector()[0]
    if params.exposure_time:
        exposure = params.exposure_time
    else:
        exposure = scan.get_exposure_times()[0]

    trusted = detector.get_trusted_range()

    indices = imageset.indices()

    if params.scan_range:
        start, end = params.scan_range
        indices = indices[start:end]
    else:
        start = 0
        end = indices[-1] + 1

    counts = flex.double(len(indices))

    t0 = time.time()

    if params.remove_spots:
        from dials.algorithms.spot_finding.factory import SpotFinderFactory
        from dials.algorithms.spot_finding.factory import phil_scope
        from dxtbx import datablock

        spot_params = phil_scope.fetch(source=iotbx.phil.parse("")).extract()
        threshold_function = SpotFinderFactory.configure_threshold(
            spot_params, datablock.DataBlock([imageset])
        )
    else:
        threshold_function = None

    for i in indices:
        pixels = imageset.get_raw_data(i)[0].as_double()

        negative = pixels < 0
        hot = pixels > int(round(trusted[1]))
        bad = negative | hot

        if threshold_function:
            peak_pixels = threshold_function.compute_threshold(pixels, ~bad)
            good = pixels.select((~bad & ~peak_pixels).iselection())
        else:
            good = pixels.select((~bad).iselection())

        counts[i - start] = flex.sum(good)

    t1 = time.time()

    print("Read data for %d images in %.1fs" % (len(indices), t1 - t0))

    # scale data to give sensible FFT values

    mean = flex.sum(counts) / counts.size()
    counts /= mean

    power = fft(counts)

    f_hz = 1.0 / exposure
    f_scale = f_hz / counts.size()

    print("Sample frequency: %.2f Hz" % f_hz)
    print("Writing output to: %s" % params.output_file)

    fout = open(params.output_file, "w")
    for j in range(power.size()):
        fout.write("%f %f\n" % (j * f_scale, power[j]))
    fout.close()


def main():
    from dials.util.options import OptionParser
    from dials.util.options import flatten_datablocks

    usage = "%prog [options] image_*.cbf"

    parser = OptionParser(
        usage=usage,
        phil=phil_scope,
        read_datablocks=True,
        read_datablocks_from_images=True,
        epilog=help_message,
    )

    params, options = parser.parse_args(show_diff_phil=True)
    datablocks = flatten_datablocks(params.input.datablock)

    if len(datablocks) == 0:
        parser.print_help()
        exit()

    datablock = datablocks[0]
    imageset = datablock.extract_imagesets()[0]
    stability_fft(imageset, params)


if __name__ == "__main__":
    main()
