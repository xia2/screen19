from generator import Generator
from watcher import Watcher
import mock
import itertools
import time

def test_instantiate_watcher():
  g = itertools.count(0)
  cb = mock.Mock()
  w = Watcher(files=g, callback=cb)
  assert not cb.called
  assert next(g) == 0

@mock.patch('watcher.os')
@mock.patch('watcher.time')
def test_start_asynchronous_watcher(mocktime, mockos):
  mockos.path.file_exists.return_value = True

  files = list(Generator('somefile_%05d.cbf', 1, 10))
  cb = mock.Mock()
  w = Watcher(files=iter(files), callback=cb, basespeed=3)

  w.start(asynchronous=False)

  assert w.is_active() == False
  assert mockos.path.exists.call_count == len(files)
  assert mocktime.sleep.call_count == 0
  assert cb.call_count == len(files)
  for call, f in zip(cb.call_args_list, files):
    args, kwargs = call
    assert args == ()
    assert kwargs == { 'file': f }
