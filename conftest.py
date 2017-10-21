from __future__ import absolute_import, division, print_function

def pytest_collect_file(path, parent):
  '''
  libtbx compatibility layer:

  collect all libtbx legacy tests to run them from within pytest.
  '''

  import os
  if 'LIBTBX_SKIP_PYTEST' in os.environ:
    '''The pytest discovery process is ran from within libtbx, so do not
       attempt to find libtbx legacy tests.'''
    return

  import libtbx
  from libtbx.test_utils.parallel import make_commands, run_command
  import pytest
  import sys

  class LibtbxTestException(Exception):
    ''' custom exception for error reporting '''
    def __init__(self, stdout, stderr):
      self.stdout = stdout
      self.stderr = stderr

  class LibtbxTest(pytest.Item):
    def __init__(self, name, parent, test_command, test_parameters):
      super(LibtbxTest, self).__init__(name, parent)
      self.test_cmd = make_commands([test_command])[0]
      self.test_parms = test_parameters
      self.full_cmd = " ".join([self.test_cmd] + self.test_parms)

    def runtest(self):
      rc = run_command(self.full_cmd)
      assert rc is not None
      if rc.error_lines or rc.return_code != 0:
        raise LibtbxTestException(rc.stdout_lines, rc.stderr_lines)

    def repr_failure(self, excinfo):
      """ called when self.runtest() raises an exception. """
      if isinstance(excinfo.value, LibtbxTestException):
        return "\n".join(excinfo.value.stderr)

    def reportinfo(self):
      return self.fspath, 0, self.full_cmd

  class LibtbxRunTestsFile(pytest.File):
    def collect(self):
      try:
        os.environ['LIBTBX_SKIP_PYTEST'] = '1'
        from . import run_tests
      finally:
        del os.environ['LIBTBX_SKIP_PYTEST']

      for test in run_tests.tst_list:
        if isinstance(test, basestring):
          testfile = test
          testparams = []
          testname = 'main'
        else:
          testfile = test[0]
          testparams = [str(s) for s in test[1:]]
          testname = "_".join(str(p) for p in testparams)

        full_command = testfile.replace("$D", str(self.session.fspath))
        shortpath = testfile.replace("$D/", "")
        pytest_file_object = pytest.File(shortpath, self.session)
        yield LibtbxTest(testname, pytest_file_object, full_command, testparams)

  if path.basename == 'run_tests.py':
    return LibtbxRunTestsFile(path, parent)
