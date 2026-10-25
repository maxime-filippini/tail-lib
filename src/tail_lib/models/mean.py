"""Models of the mean of the returns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tail_lib.graph import Cache
from tail_lib.graph import Preprocessor
from tail_lib.time_series import TimeSeries

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray
    from tail_lib.graph import Dependencies
    from tail_lib.graph import Settings


class MeanModel(Preprocessor):
    """A model of the mean of the returns, as of each date.

    The mean on date `t` is the forecast of the mean of the return of date
    `t + 1`, using data up to and including `t`.
    """


class ZeroMean(MeanModel):
    """A mean of zero on every date.

    The mean on date `t` is `0`, for every series. It uses no returns.
    """

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {}

    def settings(self) -> Settings:  # noqa: D102
        return {}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        return self.compute(cache.returns)

    def compute(self, returns: TimeSeries) -> TimeSeries:
        """Build the zero means on the dates of the returns.

        Args:
            returns (TimeSeries): The returns time series. Only its dates and
                shape are used.

        Returns:
            TimeSeries: Zeros, with the dates and shape of `returns`.

        """
        return TimeSeries(
            dates=returns.dates,
            values=np.zeros(returns.values.shape),
        )


class ConstantMean(MeanModel):
    """The same mean on every date.

    The mean on date `t` is `mean`, for every `t`. It uses no returns.
    """

    def __init__(self, mean: FloatArray | float) -> None:
        """Build the model.

        Args:
            mean (FloatArray | float): The mean, a single value or an array
                broadcastable to the batch shape of the returns.

        """
        self._mean = np.asarray(mean)

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {}

    def settings(self) -> Settings:  # noqa: D102
        return {"mean": self._mean}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        return self.compute(cache.returns)

    def compute(self, returns: TimeSeries) -> TimeSeries:
        """Repeat the mean on every date of the returns.

        Args:
            returns (TimeSeries): The returns time series. Only its dates and
                shape are used.

        Returns:
            TimeSeries: The mean, with the dates and shape of `returns`.

        """
        return TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(self._mean, returns.values.shape).copy(),
        )
