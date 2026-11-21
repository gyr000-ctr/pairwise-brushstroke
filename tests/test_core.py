import math

import pytest

import brushstroke

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PTS = [((0.0, 0.0), 0.0), ((1.0, 0.0), 1.0)]


def _b():
    return brushstroke.Brush(0.25)


def _check_nonempty():
    return len(_b().stroke(PTS)) > 0


def _check_spacing():
    out = _b().stroke(PTS)
    return all(abs(p[0][0] - 0.0) <= 1.0 for p in out)


def _check_pressure_end():
    out = _b().stroke(PTS)
    return abs(out[-1][1] - 1.0) < 1e-9


def _check_pressure_start():
    out = _b().stroke(PTS)
    return abs(out[0][1]) < 1e-9


def _check_monotonic():
    out = _b().stroke(PTS)
    return all(out[i][1] <= out[i + 1][1] + 1e-9 for i in range(len(out) - 1))


def _check_bad_points():
    return _raises(lambda: _b().stroke([((0.0, 0.0), 0.0)]))


def _check_bad_spacing():
    return _raises(lambda: brushstroke.Brush(0.0))


def _check_empty():
    return _raises(lambda: _b().stroke([]))


def _check_type():
    return isinstance(_b().stroke(PTS), list)


def _check_det():
    return _b().stroke(PTS) == _b().stroke(PTS)


def _check_count():
    return len(_b().stroke(PTS)) >= 2


def _check_width():
    w = _b().width_at(0.5)
    return 0.0 < w < 1.0 or w > 0.0


def _check_pressure_range():
    return all(0.0 <= p[1] <= 1.0 for p in _b().stroke(PTS))


def _check_no_mutation():
    src = list(PTS)
    _b().stroke(PTS)
    return PTS == src


def _check_two_segments():
    pts = [((0.0, 0.0), 0.0), ((1.0, 0.0), 0.5), ((2.0, 0.0), 1.0)]
    return len(_b().stroke(pts)) >= 3

CASES = [("nonempty", "_check_nonempty"), ("spacing", "_check_spacing"), ("pressure_end", "_check_pressure_end"), ("pressure_start", "_check_pressure_start"), ("monotonic", "_check_monotonic"), ("bad_points", "_check_bad_points"), ("bad_spacing", "_check_bad_spacing"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("count", "_check_count"), ("width", "_check_width"), ("pressure_range", "_check_pressure_range"), ("no_mutation", "_check_no_mutation"), ("two_segments", "_check_two_segments")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
