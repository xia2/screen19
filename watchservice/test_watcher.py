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
  mockos.path.exists.return_value = True

  files = list(Generator('somefile_%05d.cbf', 1, 10))
  cb = mock.Mock()
  w = Watcher(files=iter(files), callback=cb, basespeed=3)

  w.start(asynchronous=False)

  assert w.is_active() == False

  assert mockos.path.exists.call_count == len(files)
  for call, f in zip(mockos.path.exists.call_args_list, files):
    args, kwargs = call
    assert args == (f, )
    assert kwargs == {}

  assert mocktime.sleep.call_count == 0

  assert cb.call_count == len(files)
  for call, f in zip(cb.call_args_list, files):
    args, kwargs = call
    assert args == ()
    assert kwargs == { 'file': f, 'success': True, 'wait': 0 }


@mock.patch('watcher.os')
@mock.patch('watcher.time')
def test_watcher_backoff_strategy(mocktime, mockos):
  mockos.path.exists.return_value = False

  files = list(Generator('somefile_%05d.cbf', 1, 10))
  cb = mock.Mock()
  base = 3
  w = Watcher(files=iter(files), callback=cb, basespeed=base)

  w.start(asynchronous=False)

  assert w.is_active() == False
  assert mockos.path.exists.call_count > 1
  assert mockos.path.exists.call_count < 30
  for call in mockos.path.exists.call_args_list:
    args, kwargs = call
    assert args == ( files[0], )
    assert kwargs == {}

  assert mocktime.sleep.call_count == mockos.path.exists.call_count - 1
  npow = 1
  for call in mocktime.sleep.call_args_list:
    args, kwargs = call
    assert len(args) == 1
    assert kwargs == {}
    t = args[0]
    assert t >= base
    assert t <= base * npow
    assert npow * base <= 4096
    npow = 2 * npow
  assert t == 2048

  assert cb.call_count == 1
  args, kwargs = cb.call_args
  assert args == ()
  assert kwargs['wait'] > 0
  del kwargs['wait']
  assert kwargs == { 'file': files[0], 'success': False }
