from __future__ import division
from logging import info, debug, warn
import math
import re
import sys

from i19.util.procrunner import run_process

help_message = '''

This program processes screening data obtained at Diamond Light Source
Beamline I19-1.

Examples::

  i19.screen datablock.json

  i19.screen *.cbf

  i19.screen /path/to/data/

'''

class i19Screen():
  import libtbx.load_env
  from libtbx.utils import Sorry

  def _prettyprint_dictionary(self, d):
    return "{\n%s\n}" % \
      "\n".join(["  %s: %s" % (k, str(d[k]).replace("\n", "\n%s" % (" " * (4 + len(k)))))
        for k in d.iterkeys() ])

  def _import(self, files):
    info("\nImporting data...")
    command = [ "dials.import" ] + files
    debug("running %s" % " ".join(command))

    result = run_process(command, print_stdout=False)
    debug("result = %s" % self._prettyprint_dictionary(result))

    if result['exitcode'] == 0:
      info("Successfully completed (%.1f sec)" % result['runtime'])
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _count_processors(self):
    command = [ "libtbx.show_number_of_processors" ]
    debug("running %s" % command)
    result = run_process(command, print_stdout=False)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      self.nproc = result['stdout']
    else:
      warn("Could not determine number of available processors. Error code %d" % result['exitcode'])
      sys.exit(1)

  def _check_intensities(self):
    info("\nTesting pixel intensities...")
    command = [ "xia2.overload", self.json_file ]
    result = run_process(command, print_stdout=False)
    debug("result = %s" % self._prettyprint_dictionary(result))

    if result['exitcode'] == 0:
      print "Pixel intensity distribution:"
      hist = {}
      for l in result['stdout'].split("\n"):
        m = re.search('^([0-9.]+) - [0-9.]+: ([0-9]+)$', l)
        if m and m.group(2) != '0': # and m.group(1) != '0' and m.group(2) != '0':
          hist[float(m.group(1))] = int(m.group(2))
      histcount = sum(hist.itervalues())
      del hist[0]

      # There is a possibility that _sigma_m should be doubled here
      scale = 1 / (math.sqrt(math.pi) * self._sigma_m * math.erf(self._oscillation / 2 / self._sigma_m))
      info("Determined scale factor for intensities as %f" % scale)
      hist_corrected = { x*scale: hist[x] for x in hist.iterkeys() }

      self._plot_intensities(hist_corrected)

      if (histcount % self._num_images) != 0:
        warn("Warning: There may be undetected overloads above the upper bound!")

      info("Successfully completed (%.1f sec)" % result['runtime'])
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _plot_intensities(self, bins):
    import subprocess
    rows, columns = subprocess.check_output(['stty', 'size']).split()

    command = [ "gnuplot" ]
    plot_commands = [
      "set term dumb %s %d" % (columns, int(rows)-2),
      "set title 'Spot intensity distribution'",
      "set xlabel '% of maximum'",
      "set ylabel 'Number of observed pixels'",
      "set boxwidth 1.0",
      "set xtics out nomirror",
      "plot '-' using 1:2 title '' with boxes"
    ]
    for x in sorted(bins.iterkeys()):
      plot_commands.append("%f %d" % (x, bins[x]))
    plot_commands.append("e")

    debug("running %s with:\n  %s\n" % (" ".join(command), "\n  ".join(plot_commands)))

    result = run_process(command, stdin="\n".join(plot_commands)+"\n", timeout=120, print_stdout=False)

    debug("result = %s" % self._prettyprint_dictionary(result))

    if result['exitcode'] == 0:
      star = re.compile(r'\*')
      space = re.compile(' ')
      state = set()
      for l in result['stdout'].split("\n"):
        if l.strip() != '':
          stars = {m.start(0) for m in re.finditer(star, l)}
          if len(stars) == 0:
            state = set()
          else:
            state |= stars
            l = list(l)
            for s in state: l[s] = '*'
          info("".join(l))
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _find_spots(self):
    info("\nSpot finding...")
    command = [ "dials.find_spots", self.json_file, "nproc=%s" % self.nproc ]
    result = run_process(command, print_stdout=False)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      m = re.search('Saved ([0-9]+) reflections to strong.pickle', result['stdout'])
      info("Found %s reflections (%.1f sec)" % (m.group(1), result['runtime']))
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _index(self):
    info("\nIndexing...")
    command = [ "dials.index", self.json_file, "strong.pickle" ]
    result = run_process(command, print_stdout=False)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      m = re.search('model [0-9]+ \(([0-9]+) [^\n]*\n[^\n]*\n[^\n]*Unit cell: \(([^\n]*)\)\n[^\n]*Space group: ([^\n]*)\n', result['stdout'])
      info("Found primitive solution: %s (%s) using %s reflections" % (m.group(3), m.group(2), m.group(1)))
      info("Successfully completed (%.1f sec)" % result['runtime'])
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _create_profile_model(self):
    info("\nCreating profile model...")
    command = [ "dials.create_profile_model", "experiments.json", "indexed.pickle" ]
    result = run_process(command, print_stdout=False)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      from dxtbx.model.experiment.experiment_list import ExperimentListFactory
      db = ExperimentListFactory.from_json_file('experiments_with_profile_model.json')[0]
      self._num_images = db.imageset.get_scan().get_num_images()
      self._oscillation = db.imageset.get_scan().get_oscillation()[1]
      self._sigma_m = db.profile.sigma_m()
      info("%d images, %s deg. oscillation, sigma_m=%.3f" % (self._num_images, str(self._oscillation), self._sigma_m))
      info("Successfully completed (%.1f sec)" % result['runtime'])
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def _refine_bravais(self):
    info("\nRefining bravais settings...")
    command = [ "dials.refine_bravais_settings", "experiments.json", "indexed.pickle" ]
    result = run_process(command, print_stdout=False)
    debug("result = %s" % self._prettyprint_dictionary(result))
    if result['exitcode'] == 0:
      m = re.search('---+\n[^\n]*\n---+\n(.*\n)*---+', result['stdout'])
      info(m.group(0))
      info("Successfully completed (%.1f sec)" % result['runtime'])
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  def run(self, args):
    if len(args) == 0:
      print help_message
      return

    # Configure the logging
    from dials.util import log
    log.config(1, info='i19.screen.log', debug='i19.screen.debug.log')

    from dials.util.version import dials_version
    from i19.util.version import i19_version
    info("%s using %s" % (i19_version(), dials_version()))

    self._count_processors()

    if len(args) == 1 and args[0].endswith('.json'):
      self.json_file = args[0]
    else:
      self._import(args)
      self.json_file = 'datablock.json'

    self._find_spots()
    self._index()
    self._create_profile_model()
    self._check_intensities()
    self._refine_bravais()

    return

if __name__ == '__main__':
  i19Screen().run(sys.argv[1:])
