from __future__ import absolute_import, division, print_function

import itertools
import time

import mock
from i19.watchservice.generator import Generator
from i19.watchservice.watcher import Watcher


def test_instantiate_watcher():
    g = itertools.count(0)
    cb = mock.Mock()

    w = Watcher(files=g, callback=cb)

    assert not cb.called
    assert next(g) == 0


@mock.patch("i19.watchservice.watcher.os")
@mock.patch("i19.watchservice.watcher.time")
def test_start_asynchronous_watcher(mocktime, mockos):
    mockos.path.exists.return_value = True

    files = list(Generator("somefile_%05d.cbf", 1, 10))
    cb = mock.Mock()
    w = Watcher(files=iter(files), callback=cb, basespeed=3)

    w.start(asynchronous=False)

    assert w.is_active() == False

    assert mockos.path.exists.call_count == len(files)
    for call, f in zip(mockos.path.exists.call_args_list, files):
        args, kwargs = call
        assert args == (f,)
        assert kwargs == {}

    assert mocktime.sleep.call_count == 0

    assert cb.call_count == len(files)
    for call, f in zip(cb.call_args_list, files):
        args, kwargs = call
        assert args == ()
        assert kwargs == {"file": f, "success": True, "wait": 0}


@mock.patch("i19.watchservice.watcher.os")
@mock.patch("i19.watchservice.watcher.time")
def test_watcher_backoff_strategy_first_file(mocktime, mockos):
    mockos.path.exists.return_value = False

    files = list(Generator("somefile_%05d.cbf", 1, 10))
    cb = mock.Mock()
    base = 3
    w = Watcher(files=iter(files), callback=cb, basespeed=base)

    w.start(asynchronous=False)

    assert w.is_active() == False
    assert mockos.path.exists.call_count > 20
    for call in mockos.path.exists.call_args_list:
        args, kwargs = call
        assert args == (files[0],)
        assert kwargs == {}

    assert mocktime.sleep.call_count == mockos.path.exists.call_count - 1
    npow = 1
    for call in mocktime.sleep.call_args_list:
        args, kwargs = call
        assert len(args) == 1
        assert kwargs == {}
        t = args[0]
        assert t >= base
        assert t <= 3

    assert cb.call_count == 1
    args, kwargs = cb.call_args
    assert args == ()
    assert kwargs["wait"] > 0
    del kwargs["wait"]
    assert kwargs == {"file": files[0], "success": False}
