import datetime

import numpy as np
import pytest

from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tail_lib.windows import Windows

FIRST_DATE = datetime.date(2020, 1, 1)


def _series(n: int, n_series: int | None = 2) -> TimeSeries:
    shape = (n,) if n_series is None else (n, n_series)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return TimeSeries.from_first_date(FIRST_DATE, values=values)


@pytest.mark.parametrize(("length", "step"), [(0, 1), (3, 0)])
def test_trailing_rejects_empty_length_or_step(length: int, step: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Trailing(length=length, step=step)


def test_windows_values_are_time_first_then_windows() -> None:
    series = _series(6)
    v = series.values
    n_windows = 6 - 3 + 1

    windows = Windows(series, Trailing(length=3))

    assert windows.n_windows == n_windows
    assert windows.values.shape == (3, n_windows, 2)
    for k in range(n_windows):
        np.testing.assert_array_equal(windows.values[:, k], v[k : k + 3])


def test_windows_with_step_end_every_step_dates() -> None:
    series = _series(7)
    v, d = series.values, series.dates

    windows = Windows(series, Trailing(length=3, step=2))

    np.testing.assert_array_equal(windows.end_dates, d[[2, 4, 6]])
    np.testing.assert_array_equal(windows.values[:, 1], v[2:5])
    np.testing.assert_array_equal(windows.dates[:, 1], d[2:5])


def test_windows_of_one_dimensional_series() -> None:
    series = _series(5, n_series=None)

    windows = Windows(series, Trailing(length=2))

    assert windows.values.shape == (2, 4)
    np.testing.assert_array_equal(windows.values[:, 3], series.values[3:5])


def test_windows_reject_window_longer_than_series() -> None:
    with pytest.raises(ValueError, match="longer"):
        Windows(_series(3), Trailing(length=4))


def test_to_series_holds_each_result_until_the_next_window() -> None:
    series = _series(8)
    windows = Windows(series, Trailing(length=3, step=2))  # ends on 2, 4, 6

    out = windows.to_series(np.array([10.0, 20.0, 30.0]))

    np.testing.assert_array_equal(out.dates, series.dates)
    np.testing.assert_array_equal(out.values, [np.nan, np.nan, 10, 10, 20, 20, 30, 30])


def test_to_series_rejects_wrong_number_of_results() -> None:
    windows = Windows(_series(6), Trailing(length=3))

    with pytest.raises(ValueError, match="one result per window"):
        windows.to_series(np.zeros(3))


def test_apply_reduces_over_the_dates_of_each_window() -> None:
    series = _series(6)
    v = series.values

    means = Windows(series, Trailing(length=3)).apply(lambda w: w.mean(axis=0))

    assert means.values.shape == (6, 2)
    assert np.isnan(means.values[:2]).all()
    for t in range(2, 6):
        np.testing.assert_allclose(means.values[t], v[t - 2 : t + 1].mean(axis=0))
