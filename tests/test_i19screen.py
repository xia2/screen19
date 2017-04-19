from __future__ import division, absolute_import

from i19.command_line.screen import i19_screen

def test_i19screen_command_line_help_does_not_crash():
  i19_screen().run('')

def test_i19screen(tmpdir):
  import os
  import libtbx
  xia2_regression = libtbx.env.under_build("xia2_regression") 
  data_dir = os.path.join(xia2_regression, "test_data", "X4_wide")
  olddir = tmpdir.chdir()

  i19_screen().run([data_dir])

  with tmpdir.join('i19.screen.log').open() as fh:
    logfile = fh.read()

  assert 'i19.screen successfully completed' in logfile
  assert 'photon incidence rate is outside the linear response region' in logfile
