from __future__ import absolute_import, division, print_function

import pytest
from i19.command_line.screen import i19_screen

def test_i19screen_command_line_help_does_not_crash():
  i19_screen().run('')

def test_i19screen(regression_data, run_in_tmpdir):
  data_dir = regression_data('X4_wide').strpath

  i19_screen().run([data_dir])

  logfile = run_in_tmpdir.join('i19.screen.log').read()

  assert 'i19.screen successfully completed' in logfile
  assert 'photon incidence rate is outside the linear response region' in logfile
