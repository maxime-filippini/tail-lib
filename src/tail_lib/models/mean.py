"""Models of the mean of the returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tail_lib.graph import Node
from tail_lib.graph import frozen_floats
from tail_lib.time_series.time_series import TimeSeries

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray
    from tail_lib._typing import Floats


class MeanModel(Node[TimeSeries]):
    """A model of the mean of the returns, as of each date.

    The mean on date `t` is the forecast of the mean of the return of date
    `t + 1`, using data up to and including `t`.
    """


@dataclass(frozen=True)
class ZeroMean(MeanModel):
    """A mean of zero on every date.

    The mean on date `t` is `0`, for every series. It uses no returns.
    """

    def compute(self, returns: TimeSeries, /) -> TimeSeries:
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


@dataclass(frozen=True)
class ConstantMean(MeanModel):
    """The same mean on every date.

    The mean on date `t` is `mean`, for every `t`. It uses no returns.

    Attributes:
        mean (FloatArray | Floats | float): The mean, a single value or an
            array broadcastable to the batch shape of the returns. An array
            is stored as nested tuples, see `frozen_floats`.

    """

    mean: FloatArray | Floats | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", frozen_floats(self.mean))
        super().__post_init__()

    def compute(self, returns: TimeSeries, /) -> TimeSeries:
        """Repeat the mean on every date of the returns.

        Args:
            returns (TimeSeries): The returns time series. Only its dates and
                shape are used.

        Returns:
            TimeSeries: The mean, with the dates and shape of `returns`.

        """
        return TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(np.asarray(self.mean), returns.values.shape).copy(),
        )
