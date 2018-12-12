from __future__ import absolute_import, division, print_function

import sys

import mock
import pytest
from i19.command_line.screen import I19Screen

def test_i19screen_command_line_help_does_not_crash():
  sys.argv = ['i19.screen']
  I19Screen().run()

def test_i19screen(regression_data, run_in_tmpdir):
  data_dir = regression_data('X4_wide').strpath

  I19Screen().run([data_dir])

  logfile = run_in_tmpdir.join('i19.screen.log').read()

  assert 'i19.screen successfully completed' in logfile
  assert 'photon incidence rate is outside the linear response region' in logfile

@mock.patch('i19.command_line.screen.procrunner')
def test_i19screen_calls(procrunner, run_in_tmpdir):
  procrunner.run.return_value = {'exitcode': 0, 'runtime': 0}
  files = 'dataset.cbf:1:100'

  with pytest.raises(SystemExit):
    sys.argv = ['i19.screen', files]
    I19Screen().run()

  procrunner.run.assert_called_once_with([
          'dials.import',
          'input.template=dataset.cbf',
          'geometry.scan.image_range=1,100',
          'geometry.scan.extrapolate_scan=True'
      ],
      debug=False,
      print_stdout=False,
  )
