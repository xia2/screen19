from __future__ import absolute_import, division, print_function

import i19.util.time_analysis
from scitbx.array_family import flex


def test_time_analysis():
    sample_rate_hz = [100, 200, 500, 100, 50]
    length = [10000, 10000, 10000, 1000, 2000]
    f_hz = [5, 10, 20, 20, 4]
    for l, s, f in zip(length, sample_rate_hz, f_hz):
        f_scale = s / l
        d = i19.util.time_analysis.source(length=l, sample_rate_hz=s, f_hz=f)
        d = i19.util.time_analysis.fft(d)
        assert flex.max(d) == d[int(round(f / f_scale))]
