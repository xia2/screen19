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

def _prettyprint_dictionary(d):
  return "{\n%s\n}" % \
    "\n".join(["  %s: %s" % (k, str(d[k]).replace("\n", "\n%s" % (" " * (4 + len(k)))))
      for k in d.iterkeys() ])

def run(args):
  if len(args) == 0:
    print help_message
    sys.exit(0)

  import libtbx.load_env
  from libtbx.utils import Sorry
  from dials.util import log
  from logging import info, debug, warn

  # Configure the logging
  log.config(1, info='i19.screen.log', debug='i19.screen.debug.log')

  from dials.util.version import dials_version
  from i19.util.version import i19_version
  info("%s using %s" % (i19_version(), dials_version()))

  if len(args) != 1 or not args[0].endswith('.json'):
    info("\nImporting data...")
    command = [ "dials.import" ] + args
    debug("running %s" % " ".join(command))

    result = run_process(command, print_stdout=False)
    debug("result = %s" % _prettyprint_dictionary(result))

    if result['exitcode'] == 0:
      info("Successfully completed (%.1f sec)" % result['runtime'])
    else:
      warn("Failed with exit code %d" % result['exitcode'])
      sys.exit(1)

  info("\nTesting pixel intensities...")

  info("\nSpot finding...")

  info("\nIndexing...")

  info("\nRefining bravais settings...")

  return


if __name__ == '__main__':
  import sys
  run(sys.argv[1:])
