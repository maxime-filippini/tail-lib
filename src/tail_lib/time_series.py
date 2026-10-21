"""Value-at-risk models."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    import datetime

    from tail_lib._typing import DateArray1D
    from tail_lib._typing import FloatArrayAtLeast1D


class TimeSeries:
    """A time series."""

    def __init__(
        self,
        dates: DateArray1D,
        values: FloatArrayAtLeast1D,
    ) -> None:
        self._dates = dates
        self._values = values

        if self._dates.shape[0] != self._values.shape[0]:
            raise RuntimeError

    def __getitem__(self, key: Any) -> TimeSeries:  # noqa: ANN401
        if not isinstance(key, slice):
            msg = "Indexing into a time series can only be done with a slice!"
            raise TypeError(msg)
        return TimeSeries(dates=self._dates[key], values=self._values[key])

    def check_aligned(self, other: TimeSeries) -> None:
        """Check that another time series has the same dates and value shape.

        Args:
            other (TimeSeries): The time series to compare against.

        Raises:
            ValueError: If the value shapes or the dates differ.

        """
        if self._values.shape != other.values.shape:
            msg = (
                "Time series have different shapes: "
                f"{self._values.shape} vs {other.values.shape}!"
            )
            raise ValueError(msg)

        if not np.array_equal(self._dates, other.dates):
            msg = "Time series have different dates!"
            raise ValueError(msg)

    @property
    def n_dates(self) -> int:
        """Number of time steps in the time series.

        Returns:
            int: The number of time steps.

        """
        return self._dates.size

    @property
    def dates(self) -> DateArray1D:
        """The date array.

        Returns:
            DateArray1D: The date array.

        """
        return self._dates

    @property
    def values(self) -> FloatArrayAtLeast1D:
        """The value array.

        Returns:
            NonEmptyFloatArray: The value array.

        """
        return self._values

    @classmethod
    def from_first_date(
        cls,
        first_date: datetime.date,
        values: FloatArrayAtLeast1D,
    ) -> TimeSeries:
        """Build a time series from a starting date.

        Args:
            first_date (datetime.date): The first date of the time series.
            values (NonEmptyFloatArray): The value array.

        Returns:
            TimeSeries: Resulting time series.

        """
        np_first_date = np.datetime64(first_date, "D")
        n = values.shape[0]
        date_arr = np.arange(np_first_date, np_first_date + n)
        return cls(dates=date_arr, values=values)

    @classmethod
    def to_last_date(
        cls,
        last_date: datetime.date,
        values: FloatArrayAtLeast1D,
    ) -> TimeSeries:
        """Build a time series from an ending date.

        Args:
            last_date (datetime.date): The first date of the time series.
            values (NonEmptyFloatArray): The value array.

        Returns:
            TimeSeries: Resulting time series.

        """
        np_last_date = np.datetime64(last_date, "D")
        n = values.shape[0]
        date_arr = np.arange(np_last_date - n + 1, np_last_date + 1)
        return cls(dates=date_arr, values=values)
