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


def _with_leading_nan(n: int, n_nan: list[int]) -> TimeSeries:
    values = np.arange(n * len(n_nan), dtype=float).reshape(n, len(n_nan))
    for j, k in enumerate(n_nan):
        values[:k, j] = np.nan
    return TimeSeries.from_first_date(FIRST_DATE, values=values)


def test_windows_skipping_leading_nan_start_on_the_first_value() -> None:
    series = _with_leading_nan(10, [3])
    v, d = series.values, series.dates

    windows = Windows(series, Trailing(length=3, step=2), skip_leading_nan=True)

    # The first value is on date 3: windows end on 5, 7 and 9.
    np.testing.assert_array_equal(windows.end_dates, d[[5, 7, 9]])
    np.testing.assert_array_equal(windows.values[:, 0], v[3:6])
    np.testing.assert_array_equal(windows.dates[:, 0], d[3:6])
    assert not np.isnan(windows.values).any()


def test_windows_skipping_leading_nan_wait_for_every_series() -> None:
    series = _with_leading_nan(8, [1, 4])

    windows = Windows(series, Trailing(length=2), skip_leading_nan=True)

    np.testing.assert_array_equal(windows.values[:, 0], series.values[4:6])
    out = windows.to_series(np.arange(windows.n_windows, dtype=float))
    np.testing.assert_array_equal(out.values, [np.nan] * 5 + [0, 1, 2])


def test_windows_skipping_leading_nan_reject_too_short_a_rest() -> None:
    with pytest.raises(ValueError, match="first date where every series"):
        Windows(_with_leading_nan(6, [4]), Trailing(length=3), skip_leading_nan=True)

    with pytest.raises(ValueError, match="longer"):
        Windows(_with_leading_nan(3, [3]), Trailing(length=1), skip_leading_nan=True)


def test_windows_keep_leading_nan_by_default() -> None:
    series = _with_leading_nan(6, [2])

    windows = Windows(series, Trailing(length=3))

    assert windows.end_dates[0] == series.dates[2]
    assert np.isnan(windows.values[0, 0]).all()


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
