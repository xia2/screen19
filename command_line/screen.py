from __future__ import division
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

from logging import info, debug, warn

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
      for l in result['stdout'].split("\n"):
        if not l.endswith(': 0') and not l == "":
          info(l)
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
      import re
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
      import re
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
      import re
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

    if True:
      self._check_intensities()

    if True:
      self._find_spots()

    if True:
      self._index()

    if True:
      self._refine_bravais()

    return

if __name__ == '__main__':
  import sys
  i19Screen().run(sys.argv[1:])
