from __future__ import division
from logging import info, debug, warn
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
        m = re.search('^([0-9]+) - [0-9]+: ([0-9]+)$', l)
        if m and m.group(1) != '0' and m.group(2) != '0':
          hist[int(m.group(1))] = int(m.group(2))
      hist_maxval=max(hist.itervalues())
      hist_height=20

      hist_width=max(hist.iterkeys())
      hist_xlab=1
      if hist_width > 9: hist_xlab=2
      if hist_width > 99: hist_xlab=3

      info("")
      info(hist_maxval)
      hist_scaled={x: round(hist_height * hist[x] / hist_maxval, 1) for x in hist.iterkeys()}
      for l in range(hist_height):
        line = {x: (2 if (hist_height - l - 0.3) < hist_scaled[x] 
              else (1 if (hist_height - l - 0.8) < hist_scaled[x] or (l == hist_height - 1)
              else  0)) for x in hist.iterkeys()}
        s = ''
        for x in range(hist_width):
          if (x+1) in line:
            if line[x+1] == 2:
              s+='#'
            elif line[x+1] == 1:
              s+='\033[90m#\033[0m'
            else:
              s+=' '
          else:
            s+=' '
        info('|%s' % s)
      info('-%s' % (hist_width * '-'))

      xlabel = [      
       ' ' + (' ' * 99) + ('1' * 100) + ('2' * 100) + ('3' * 100) + ('4' * 100) + '5',
       ' ' + (' ' * 9) + (('1' * 10) + ('2' * 10) + ('3' * 10) + ('4' * 10) + ('5' * 10)
           + ('6' * 10) + ('7' * 10) + ('8' * 10) + ('9' * 10) + ('0' * 10)) * 5,
       ' ' + ('1234567890' * 50)
      ]
      for l in xlabel:
        if l[:hist_width+1].strip() != '':
          info(l[:hist_width+1])

      info("")
      info("Successfully completed (%.1f sec)" % result['runtime'])
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

    self._check_intensities()
    self._find_spots()
    self._index()
    self._refine_bravais()

    return

if __name__ == '__main__':
  i19Screen().run(sys.argv[1:])
