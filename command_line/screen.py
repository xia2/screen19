from __future__ import division
from logging import info, debug, warn
import json
import math
import os
import re
import sys
import time

from dials.util.procrunner import run_process

help_message = '''
This program processes screening data obtained at Diamond Light Source
Beamline I19-1.

Examples:

  i19.screen datablock.json

  i19.screen *.cbf

  i19.screen /path/to/data/

'''

procrunner_debug = True

class i19_screen():
  import libtbx.load_env

  def _prettyprint_dictionary(self, d):
    return "{\n%s\n}" % \
      "\n".join(["  %s: %s" % (k, str(d[k]).replace("\n", "\n%s" % (" " * (4 + len(k)))))
        for k in d.iterkeys() ])

  def _quick_import(self, files):
    debug("Attempting quick import...")
    files.sort()
    template = None
    templates = []
    for f in files:
      if template is None:
        template = { 't': f, 'count': 1 }
        continue
      if len(template['t']) != len(f):
        templates.append(template)
        template = { 't': f, 'count': 1 }
        continue
      # Find positions where file names differ
      template_positions = filter(lambda x: x is not None, \
                             map(lambda (x,y,z):z if x!=y else None, \
                               zip(template['t'], f, range(len(f)))))
      template_positions = (min(template_positions), max(template_positions))
      # This must not conflict with previously determined template information
      if 'range' in template:
        if (template_positions[0]+1 < template['range'][0]) \
        or (template_positions[0]   > template['range'][1]) \
        or (template_positions[1]   > template['range'][1]):
          templates.append(template)
          template = { 't': f, 'count': 1 }
          continue
        template_positions = (min(template_positions[0], template['range'][0]), template['range'][1])
      # Check if filename can be predicted using existing template information
      predicted_filename = template['t'][:template_positions[0]] + \
                           ("%%0%dd" % (1 + template_positions[1] - template_positions[0]) % (
                             int(template['t'][template_positions[0]:1 + template_positions[1]]) + \
                             template['count'])) + \
                           template['t'][1 + template_positions[1]:]
      if f != predicted_filename:
        templates.append(template)
        template = { 't': f, 'count': 1 }
        continue
      template['range'] = template_positions
      template['count'] = 1 + (template['count'] if 'count' in template else 0)
    templates.append(template)
    debug("Quick import template summary:")
    debug(templates)

    if len(templates) > 1:
      debug("Cannot currently run quick import on multiple templates")
      return False

    info("Running quick import")

    scan_range = int(templates[0]['t'][templates[0]['range'][0]:templates[0]['range'][1]+1])
    scan_range = (scan_range, scan_range + templates[0]['count'] - 1)

    self._run_dials_import([templates[0]['t'], "geometry.scan.image_range=%d,%d" % scan_range, "geometry.scan.extrapolate_scan=True"])

    return True

  def _import(self, files):
    info("\nImporting data...")
    if len(files) == 1 and os.path.isdir(files[0]):
      debug("You specified a directory. Importing all CBF files in that directory.")
      files = [ os.path.join(files[0], f) for f in os.listdir(files[0]) if f.endswith('.cbf') ]

    # Can the files be quick-imported?
    if self._quick_import(files):
      info("Quick import successful")
      return

    self._run_dials_import(files)

  def _run_dials_import(self, parameters):
    command = [ "dials.import" ] + parameters
    debug("running %s" % " ".join(command))

    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))

    if result['exitcode'] == 0:
      if os.path.isfile('datablock.json'):
        info("Successfully completed (%.1f sec)" % result['runtime'])
      else:
        warn("Could not import images. Do the specified images exist at that location?")
        sys.exit(1)
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _count_processors(self, nproc=None):
    if nproc is not None:
      self.nproc = nproc
      return
    command = [ "libtbx.show_number_of_processors" ]
    debug("running %s" % command)
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      self.nproc = result['stdout'].strip()
    else:
      warn("Could not determine number of available processors. Error code %d" % result['exitcode'])
      sys.exit(1)

  def _check_intensities(self):
    info("\nTesting pixel intensities...")
    command = [ "xia2.overload", self.json_file ]
    debug("running %s" % command)
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))

    if result['exitcode'] != 0:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

    with open('overload.json') as fh:
      overload_data = json.load(fh)

    print "Pixel intensity distribution:"
    count_sum = 0
    hist = {}
    for b in range(overload_data['bin_count']):
      if overload_data['bins'][b] > 0:
        hist[b] = overload_data['bins'][b]
        count_sum += b * overload_data['bins'][b]

    histcount = sum(hist.itervalues())

    # we have checked this: if _sigma_m >> _oscillation it works out about 1
    # as you would expect
    M = math.sqrt(math.pi) * self._sigma_m * \
      math.erf(self._oscillation / (2 * self._sigma_m))
    average_to_peak = M / self._oscillation

    info("Average-to-peak intensity ratio: %f" % average_to_peak)
    scale = 100 * overload_data['scale_factor'] / average_to_peak
    info("Determined scale factor for intensities as %f" % scale)
    debug("intensity histogram: { %s }", ", ".join(["%d:%d" % (k, hist[k]) for k in sorted(hist)]))
    max_count = max(hist.iterkeys())
    rescaled_hist = {}
    for x in hist.iterkeys():
      rescaled = int(x * scale)
      try:
        rescaled_hist[rescaled] += hist[x]
      except Exception:
        rescaled_hist[rescaled] = hist[x]
    hist = rescaled_hist
    debug("rescaled histogram: { %s }", ", ".join(["%d:%d" % (k, hist[k]) for k in sorted(hist)]))
    hist_max = max(hist.iterkeys())

    del hist[0]
    self._plot_intensities(hist)

    text = "Strongest pixel (%d counts) reaches %.1f %% of the detector count rate limit" % (max_count, hist_max)
    if (hist_max > 100):
      warn("Warning: %s!" % text)
    else:
      info(text)

    if (histcount % self._num_images) != 0:
      warn("Warning: There may be undetected overloads above the upper bound!")

    info("Total sum of counts in dataset: %d" % count_sum)

    info("Successfully completed (%.1f sec)" % result['runtime'])

  def _plot_intensities(self, bins):
    columns, rows = 80, 25
    if sys.stdout.isatty():
      try:
        result = run_process(['stty', 'size'], timeout=1, print_stdout=False, print_stderr=False, debug=procrunner_debug)
        rows, columns = [int(i) for i in result['stdout'].split()]
      except Exception: # ignore any errors and use default size
        pass
    rows = min(rows, int(columns / 3))

    command = [ "gnuplot" ]
    plot_commands = [
      "set term dumb %d %d" % (columns, rows-2),
      "set title 'Spot intensity distribution'",
      "set xlabel '% of maximum'",
      "set ylabel 'Number of observed pixels'",
      "set logscale y",
      "set boxwidth 1.0",
      "set xtics out nomirror",
      "set ytics out",
      "plot '-' using 1:2 title '' with boxes"
    ]
    for x in sorted(bins.iterkeys()):
      plot_commands.append("%f %d" % (x, bins[x]))
    plot_commands.append("e")

    debug("running %s with:\n  %s\n" % (" ".join(command), "\n  ".join(plot_commands)))

    result = run_process(command, stdin="\n".join(plot_commands)+"\n", timeout=120,
        print_stdout=False, print_stderr=False, debug=procrunner_debug)

    debug("result = %s" % self._prettyprint_dictionary(result))

    if result['exitcode'] == 0:
      star = re.compile(r'\*')
      space = re.compile(' ')
      state = set()
      for l in result['stdout'].split("\n"):
        if l.strip() != '':
          stars = {m.start(0) for m in re.finditer(star, l)}
          if not stars:
            state = set()
          else:
            state |= stars
            l = list(l)
            for s in state: l[s] = '*'
          info("".join(l))
    else:
      warn("Error running gnuplot. Can not plot intensity distribution. Exit code %d" % result['exitcode'])

  def _find_spots(self, additional_parameters=None):
    if additional_parameters is None:
      additional_parameters = []
    info("\nSpot finding...")
    command = [ "dials.find_spots", self.json_file, "nproc=%s" % self.nproc ] + additional_parameters
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] != 0:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)
    info(60 * '-')
    from libtbx import easy_pickle
    from dials.util.ascii_art import spot_counts_per_image_plot
    refl = easy_pickle.load('strong.pickle')
    info(spot_counts_per_image_plot(refl))
    info(60 * '-')
    info("Successfully completed (%.1f sec)" % result['runtime'])


  def _index(self):
    base_command = [ "dials.index", self.json_file, "strong.pickle", "indexing.nproc=%s" % self.nproc ]
    runlist = [
      ("Indexing...",
        base_command),
      ("Retrying with max_cell constraint",
        base_command + [ "max_cell=20" ]),
      ("Retrying with 1D FFT",
        base_command + [ "indexing.method=fft1d" ])
      ]

    for message, command in runlist:
      info("\n%s..." % message)

      result = run_process(command, print_stdout=False, debug=procrunner_debug)
      debug("result = %s" % self._prettyprint_dictionary(result))
      if result['exitcode'] != 0:
        warn("Failed with exit code %d" % result['exitcode'])
      else:
        break

    if result['exitcode'] != 0:
      return False

    m = re.search('model [0-9]+ \(([0-9]+) [^\n]*\n[^\n]*\n[^\n]*Unit cell: \(([^\n]*)\)\n[^\n]*Space group: ([^\n]*)\n', result['stdout'])
    info("Found primitive solution: %s (%s) using %s reflections" % (m.group(3), m.group(2), m.group(1)))
    info("Successfully completed (%.1f sec)" % result['runtime'])
    return True

  def _refine(self):
    info("\nIndexing...")
    command = [ "dials.refine", "experiments.json", "indexed.pickle" ]
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] != 0:
      warn("Failed with exit code %d" % result['exitcode'])
      warn("Giving up.")
      sys.exit(1)

    info("Successfully refined (%.1f sec)" % result['runtime'])
    os.rename("experiments.json", "experiments.unrefined.json")
    os.rename("indexed.pickle", "indexed.unrefined.pickle")
    os.rename("refined_experiments.json", "experiments.json")
    os.rename("refined.pickle", "indexed.pickle")

  def _predict(self):
    info("\nPredicting reflections...")
    command = [ "dials.predict", "experiments_with_profile_model.json" ]
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      info("To view predicted reflections run:")
      info("  dials.image_viewer experiments_with_profile_model.json predicted.pickle")
      info("Successfully completed (%.1f sec)" % result['runtime'])
      return True
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      return False

  def _create_profile_model(self):
    info("\nCreating profile model...")
    command = [ "dials.create_profile_model", "experiments.json", "indexed.pickle" ]
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      from dxtbx.model.experiment.experiment_list import ExperimentListFactory
      db = ExperimentListFactory.from_json_file('experiments_with_profile_model.json')[0]
      self._num_images = db.imageset.get_scan().get_num_images()
      self._oscillation = db.imageset.get_scan().get_oscillation()[1]
      self._sigma_m = db.profile.sigma_m()
      info("%d images, %s deg. oscillation, sigma_m=%.3f" % (self._num_images, str(self._oscillation), self._sigma_m))
      info("Successfully completed (%.1f sec)" % result['runtime'])
      return True
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      return False

  def _refine_bravais(self):
    info("\nRefining bravais settings...")
    command = [ "dials.refine_bravais_settings", "experiments.json", "indexed.pickle" ]
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      m = re.search('---+\n[^\n]*\n---+\n(.*\n)*---+', result['stdout'])
      info(m.group(0))
      info("Successfully completed (%.1f sec)" % result['runtime'])
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _report(self):
    info("\nCreating report...")
    command = [ "dials.report", "experiments_with_profile_model.json", "indexed.pickle" ]
    result = run_process(command, print_stdout=False, debug=procrunner_debug)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      info("Successfully completed (%.1f sec)" % result['runtime'])
#     if sys.stdout.isatty():
#       info("Trying to start browser")
#       try:
#         import subprocess
#         d = dict(os.environ)
#         d["LD_LIBRARY_PATH"] = ""
#         subprocess.Popen(["xdg-open", "dials-report.html"], env=d)
#       except Exception, e:
#         debug("Could not open browser")
#         debug(str(e))
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def run(self, args):
    from dials.util.version import dials_version
    from i19.util.version import i19_version
    version_information = "%s using %s (%s)" % (i19_version(), dials_version(), time.strftime("%Y-%m-%d %H:%M:%S"))

    if len(args) == 0:
      print help_message
      print version_information
      return

    # Configure the logging
    from dials.util import log
    log.config(1, info='i19.screen.log', debug='i19.screen.debug.log')

    info(version_information)
    debug('Run with %s' % str(args))

    # FIXME use proper optionparser here. This works for now
    nproc = None
    if len(args) >= 1 and args[0].startswith('nproc='):
      nproc = args[0][6:]
      args = args[1:]
    self._count_processors(nproc=nproc)
    debug('Using %s processors' % self.nproc)

    if len(args) == 1 and args[0].endswith('.json'):
      self.json_file = args[0]
    else:
      self._import(args)
      self.json_file = 'datablock.json'

    self._find_spots()
    if not self._index():
      info("\nRetrying for stronger spots only...")
      os.rename("strong.pickle", "all_spots.pickle")
      self._find_spots(['sigma_strong=15'])
      if not self._index():
        warn("Giving up.")
        info("""
Could not find an indexing solution. You may want to have a look
at the reciprocal space by running:

  dials.reciprocal_lattice_viewer datablock.json all_spots.pickle

or, to only include stronger spots:

  dials.reciprocal_lattice_viewer datablock.json strong.pickle
""")
        sys.exit(1)

    if not self._create_profile_model():
      info("\nRefining model to attempt to increase number of valid spots...")
      self._refine()
      if not self._create_profile_model():
        warn("Giving up.")
        info("""
The identified indexing solution may not be correct. You may want to have a look
at the reciprocal space by running:

  dials.reciprocal_lattice_viewer experiments.json indexed.pickle
""")
        sys.exit(1)
    self._report()
    self._predict()
    self._check_intensities()
    self._refine_bravais()
    debug("Finished at %s" % time.strftime("%Y-%m-%d %H:%M:%S"))

    return

if __name__ == '__main__':
  i19_screen().run(sys.argv[1:])
