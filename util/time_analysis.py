from __future__ import absolute_import, division, print_function

from scitbx.array_family import flex


def source(length=10000, sample_rate_hz=100, f_hz=10):
    import math
    import random

    data = flex.double(length, 0.0)

    for j in range(length):
        data[j] = 10 * math.sin(2 * math.pi * j * (f_hz / sample_rate_hz))
        data[j] += random.random()

    return data


def fft(d):
    from scitbx import fftpack

    f = fftpack.real_to_complex(d.size())
    _d = flex.double(f.m_real(), 0.0)
    for j in range(d.size()):
        _d[j] = d[j]
    t = f.forward(_d)
    p = flex.abs(t) ** 2

    # remove the DC component
    p[0] = 0.0

    return p
