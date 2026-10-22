"""Rolling windows over a time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import assert_never

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tail_lib.time_series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Callable

    from tail_lib._typing import DateArray1D
    from tail_lib._typing import FloatArray


@dataclass(frozen=True)
class Trailing:
    """Trailing windows of `length` dates, one every `step` dates.

    The first window ends on date `length - 1`, then one ends every `step`
    dates after it.
    """

    length: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.length < 1:
            msg = f"Window length has to be at least 1, got {self.length}!"
            raise ValueError(msg)

        if self.step < 1:
            msg = f"Window step has to be at least 1, got {self.step}!"
            raise ValueError(msg)


@dataclass(frozen=True)
class Last:
    """One window: the last `length` dates, ending on the last date."""

    length: int

    def __post_init__(self) -> None:
        if self.length < 1:
            msg = f"Window length has to be at least 1, got {self.length}!"
            raise ValueError(msg)


class Windows:
    """The windows of a time series, and how to put results back on its dates.

    The values of all windows are one array, in the library's layout: time
    first (the dates inside a window), then the windows as a batch axis, then
    the batch axes of the series. A function written for one series, reducing
    over the first axis, so works on every window at once.

    The result of a window belongs to the window's last date. `to_series` puts
    results back on the dates of the series: each date holds the result of the
    last window ending on or before it, and dates before the first window are
    `NaN`.

    With `skip_leading_nan`, the windows start on the first date where every
    series has a value, so no window holds the leading `NaN`s of a warm-up
    (R16). For `Trailing` windows, the step is counted from there. A `Last`
    window has to start on or after it.
    """

    def __init__(
        self,
        series: TimeSeries,
        spec: Trailing | Last,
        *,
        skip_leading_nan: bool = False,
    ) -> None:
        """Cut the windows of a series.

        Args:
            series (TimeSeries): The time series.
            spec (Trailing | Last): The windows to cut.
            skip_leading_nan (bool): Start the windows on the first date where
                every series has a value, rather than on the first date.

        Raises:
            ValueError: If a window is longer than the series, or than its
                part from the first date with a value for every series.

        """
        n, length = series.n_dates, spec.length
        first = _first_full_date(series) if skip_leading_nan else 0
        if n - first < length:
            msg = f"Window of {length} dates is longer than the {n - first} dates"
            if skip_leading_nan:
                msg += " from the first date where every series has a value"
            raise ValueError(msg + "!")

        # The windows are cut from `start`, one every `step` dates.
        match spec:
            case Trailing():
                start, step = first, spec.step
            case Last():
                # Only one window fits from there: the last one.
                start, step = n - length, 1
            case _:
                assert_never(spec)

        self._series_dates = series.dates
        # Index of the last date of each window.
        self._ends = np.arange(start + length - 1, n, step)

        # sliding_window_view puts the dates inside a window on the last axis:
        # move them first. These are read-only views, not copies.
        strided = sliding_window_view(series.values[start:], length, axis=0)
        self._values = np.moveaxis(strided, -1, 0)[:, ::step]
        self._dates = sliding_window_view(series.dates[start:], length).T[:, ::step]

    @property
    def n_windows(self) -> int:
        """The number of windows."""
        return self._ends.size

    @property
    def values(self) -> FloatArray:
        """The values of every window, of shape (length, n_windows, *batch)."""
        return self._values

    @property
    def dates(self) -> np.ndarray:
        """The dates inside every window, of shape (length, n_windows)."""
        return self._dates

    @property
    def end_dates(self) -> DateArray1D:
        """The last date of every window, which its result belongs to."""
        return self._series_dates[self._ends]

    def to_series(self, results: FloatArray) -> TimeSeries:
        """Put one result per window back on the dates of the series.

        Args:
            results (FloatArray): The results, with the windows on the first
                axis, of shape (n_windows, *rest).

        Returns:
            TimeSeries: The results on the dates of the series, of shape
                (n_dates, *rest). Each date holds the result of the last window
                ending on or before it, `NaN` before the first window.

        Raises:
            ValueError: If there is not one result per window.

        """
        results = np.asarray(results, dtype=float)
        if results.ndim == 0 or results.shape[0] != self.n_windows:
            msg = (
                f"Expected one result per window ({self.n_windows}), got an "
                f"array of shape {results.shape}!"
            )
            raise ValueError(msg)

        n = self._series_dates.size
        # For each date, the last window ending on or before it (-1: none yet).
        window = np.searchsorted(self._ends, np.arange(n), side="right") - 1
        has_result = window >= 0

        values = np.full((n, *results.shape[1:]), np.nan)
        values[has_result] = results[window[has_result]]
        return TimeSeries(dates=self._series_dates, values=values)

    def apply(self, fn: Callable[[FloatArray], FloatArray]) -> TimeSeries:
        """Apply a function to every window, then put the results on dates.

        Args:
            fn (Callable[[FloatArray], FloatArray]): Takes `values`, of shape
                (length, n_windows, *batch), and returns one result per window,
                of shape (n_windows, *rest).

        Returns:
            TimeSeries: The results on the dates of the series, see
                `to_series`.

        """
        return self.to_series(fn(self._values))


def _first_full_date(series: TimeSeries) -> int:
    """Find the first date where every series has a value, as an index.

    Returns the number of dates if there is none.
    """
    values = series.values.reshape(series.n_dates, -1)
    full = ~np.isnan(values).any(axis=1)
    return int(np.argmax(full)) if full.any() else series.n_dates
