# LIBTBX_PRE_DISPATCHER_INCLUDE_SH export PHENIX_GUI_ENVIRONMENT=1
# LIBTBX_PRE_DISPATCHER_INCLUDE_SH export BOOST_ADAPTBX_FPE_DEFAULT=1

from __future__ import division, print_function

import json
import sys
import threading
import time
from optparse import SUPPRESS_HELP, OptionParser

import dials.command_line.image_viewer as iv
import iotbx.phil
import stomp

phil_scope = iotbx.phil.parse("""\
image_viewer {
  brightness = 10
    .type = int
  color_scheme = grayscale rainbow heatmap *invert
    .type = choice
  show_beam_center = True
    .type = bool
  show_resolution_rings = False
    .type = bool
  show_ice_rings = False
    .type = bool
  show_ctr_mass = True
    .type = bool
  show_max_pix = True
    .type = bool
  show_all_pix = True
    .type = bool
  show_shoebox = True
    .type = bool
  show_predictions = True
    .type = bool
  show_miller_indices = False
    .type = bool
  display = *image mean variance dispersion sigma_b \
            sigma_s threshold global_threshold
    .type = choice
  nsigma_b = 6
    .type = float(value_min=0)
  nsigma_s = 3
    .type = float(value_min=0)
  global_threshold = 0
    .type = float(value_min=0)
  kernel_size = 3,3
    .type = ints(size=2, value_min=1)
  min_local = 2
    .type = int
  gain = 1
    .type = float(value_min=0)
  sum_images = 1
    .type = int(value_min=1)
    .expert_level = 2
  untrusted_polygon = None
    .multiple = True
    .type = ints(value_min=0)
  d_min = None
    .type = float(value_min=0)
}
""")

class Coordinator():
  def __init__(self):
    self._to_load = None
    self._stomp = Stomp(self)
    self._lock = threading.Lock()

  def load_file(self, filename):
    with self._lock:
      self._to_load = filename

  def open_viewer(self, filename):
    import wx
    from dials.util.options import OptionParser
    from dials.util.options import flatten_datablocks
    import libtbx.load_env
    parser = OptionParser(
      phil=phil_scope,
      read_datablocks=True,
      read_experiments=True,
      read_reflections=True,
      read_datablocks_from_images=True)
    params, options = parser.parse_args(args=[filename])
    datablocks = flatten_datablocks(params.input.datablock)
    imagesets = datablocks[0].extract_imagesets()
    self.runner = iv.Script(
      params=params.image_viewer,
      reflections=[],
      imagesets=imagesets,
      crystals=None)
    # Run the script

    self.thread = threading.Thread(target=self.runner)
    self.thread.start()

  def run(self):
    while self._to_load is None:
      time.sleep(0.1)
    with self._lock:
      self.open_viewer(self._to_load)
      self._to_load = None
    while True:
      if self._to_load:
        with self._lock:
          nextfile = self._to_load
          self._to_load = None
        if self.runner.wrapper:
          self.runner.wrapper.load_image(nextfile)
      time.sleep(0.1)


class MyListener(stomp.ConnectionListener):
  def __init__(self, coordinator):
    self._coord = coordinator

#  def on_error(self, headers, message):
#    print('received an error "%s"' % message)
  def on_message(self, headers, message):
    payload = json.loads(message)
    filename = payload['file'].encode('ascii')
    self._coord.load_file(filename)


class Stomp():
  def __init__(self, coordinator):
    self._conn = stomp.Connection([('i19-1-control', 61613)])
    self._conn.set_listener('', MyListener(coordinator))
#   self._conn.set_listener('', stomp.PrintingListener())
    self._conn.start()
    self._conn.connect('i19-filewatcher', '', wait=True)
    self._conn.subscribe('/topic/i19-image-arrived', id=1, ack='auto')

  def image_arrived(self, **kwargs):
    self._conn.send(
      body=json.dumps(kwargs),
      destination = '/topic/i19-image-arrived'
    )

if __name__ == "__main__":
  Coordinator().run()
