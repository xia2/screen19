from __future__ import division
import iotbx.phil
from scitbx.array_family import flex
import time

help_message = '''

Examples::

  i19.stability_fft image_*.cbf

'''

phil_scope = iotbx.phil.parse("""\
plot = False
  .type = bool
  .help = "Make some plots"
remove_spots = False
  .type = bool
  .help = "Remove signal pixels from images before analysis"
scan_range = None
  .help = "Scan range to use for analysis"
  .type = ints(size=2)

""", process_includes=True)

def good_pixel_total(pixels, trusted):
  negative = (pixels < 0)
  hot = (pixels > int(round(trusted[1])))
  bad = negative | hot
  good = pixels.select((~bad).iselection())
  return flex.sum(good)

def stability_fft(imageset, params):
  from i19.util.time_analysis import fft

  scan = imageset.get_scan()
  detector = imageset.get_detector()[0]
  exposure = scan.get_exposure_times()[0]
  trusted = detector.get_trusted_range()

  indices = imageset.indices()

  if params.scan_range:
    start, end = params.scan_range
    indices = indices[start:end]

  counts = flex.double(len(indices))

  t0 = time.time()

  for i in indices:
    pixels = imageset.get_raw_data(i)[0]
    counts[i] = good_pixel_total(pixels, trusted)

  t1 = time.time()

  print 'Read data for %d images in %.1fs' % (len(indices), t1 - t0)

  # scale data to give sensible FFT values

  mean = flex.sum(counts) / counts.size()
  counts /= mean

  power = fft(counts)

  f_hz = 1.0 / exposure
  f_scale = f_hz / counts.size()

  for j in range(power.size()):
    print j * f_scale, power[j]



def main():
  from dials.util.options import OptionParser
  from dials.util.options import flatten_datablocks
  import libtbx.load_env

  usage = "%s [options] image_*.cbf" % (
    libtbx.env.dispatcher_name)

  parser = OptionParser(
    usage=usage,
    phil=phil_scope,
    read_datablocks=True,
    read_datablocks_from_images=True,
    epilog=help_message)

  params, options = parser.parse_args(show_diff_phil=True)
  datablocks = flatten_datablocks(params.input.datablock)

  if len(datablocks) == 0:
    parser.print_help()
    exit()

  datablock = datablocks[0]
  imageset = datablock.extract_imagesets()[0]
  stability_fft(imageset, params)

if __name__ == '__main__':
  main()
