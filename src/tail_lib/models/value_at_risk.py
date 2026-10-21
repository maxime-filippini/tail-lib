"""Value-at-risk models."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import assert_never

import numpy as np
from scipy import stats

from tail_lib.graph import Cache
from tail_lib.graph import Node
from tail_lib.models.mean import MeanModel
from tail_lib.models.mean import ZeroMean
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tail_lib.windows import Windows

if TYPE_CHECKING:
    from numpy.lib._function_base_impl import (
        _InterpolationMethod as NumpyInterpolationMethod,
    )

    from tail_lib.models.volatility import VolatilityModel


type InterpolationMethod = Literal["linear", "below", "above"]


def _interp_to_numpy_quantile_method(
    interp: InterpolationMethod,
) -> NumpyInterpolationMethod:
    match interp:
        case "linear":
            return "linear"
        case "below":
            return "lower"
        case "above":
            return "higher"
        case _:
            assert_never(interp)


class ValueAtRiskModel(Node[TimeSeries]):
    """Base class for value-at-risk models.

    A value-at-risk model is a node whose output is the value-at-risk on each
    date. Subclasses implement `evaluate`, which gets the inputs of the model
    from a `Cache`, then computes the value-at-risk. The computation itself
    usually lives in a `compute` method, whose arguments are the inputs.
    """

    _cf: float

    def __init__(self, cf: float) -> None:
        if cf <= 0 or cf >= 1:
            msg = "Confidence level has to be in (0,1)!"
            raise ValueError(msg)

        self._cf = cf

    @property
    def cf(self) -> float:
        """The confidence level of the value-at-risk."""
        return self._cf


class HistoricalSimulationsModel(ValueAtRiskModel):
    """Value-at-risk as an empirical quantile of the returns in a trailing window.

    The value-at-risk on the last date `t` of a window is the `1 - cf`
    empirical quantile of the returns in that window, i.e. of the `length`
    returns up to and including `t`:

        VaR[t] = Q_{1 - cf}(r[t - length + 1], ..., r[t])

    It is the forecast for the return of date `t + 1`. With a `step` above 1,
    each value-at-risk is held until the next window ends. Dates before the
    first full window are `NaN`.

    The quantile interpolates between order statistics as set by
    `interpolation`: "linear" (numpy's default), "below" (the order statistic
    at or below the quantile) or "above" (the one at or above it).
    """

    _window: Trailing
    _interpolation: InterpolationMethod

    def __init__(
        self,
        *,
        cf: float,
        window: Trailing,
        interpolation: InterpolationMethod = "linear",
    ) -> None:
        super().__init__(cf=cf)
        self._window = window
        self._interpolation = interpolation

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        return self.compute(cache.returns)

    def compute(self, returns: TimeSeries) -> TimeSeries:
        """Compute the value-at-risk on each window.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            TimeSeries: The value-at-risk, on the dates of the returns.

        """
        method = _interp_to_numpy_quantile_method(self._interpolation)
        return Windows(returns, self._window).apply(
            lambda w: np.quantile(w, q=1 - self._cf, method=method, axis=0)
        )


class GaussianModel(ValueAtRiskModel):
    """Value-at-risk of normally distributed returns.

    The value-at-risk on date `t` is the `1 - cf` quantile of a normal
    distribution with the mean and volatility on date `t`:

        VaR[t] = mean[t] + vol[t] * Phi^-1(1 - cf)

    It is the forecast for the return of date `t + 1`, using data up to and
    including `t`, like its inputs.
    """

    _mean_model: MeanModel
    _vol_model: VolatilityModel

    def __init__(
        self,
        *,
        cf: float,
        vol_model: VolatilityModel,
        mean_model: MeanModel | None = None,
    ) -> None:
        super().__init__(cf=cf)
        self._vol_model = vol_model
        self._mean_model = ZeroMean() if mean_model is None else mean_model

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        means = cache.get(self._mean_model)
        vols = cache.get(self._vol_model)
        return self.compute(means=means, vols=vols)

    def compute(self, *, means: TimeSeries, vols: TimeSeries) -> TimeSeries:
        """Compute the value-at-risk on each date.

        Args:
            means (TimeSeries): The means of the returns.
            vols (TimeSeries): The volatilities of the returns, with the same
                dates and shape as `means`.

        Returns:
            TimeSeries: The value-at-risk, on the dates of `means` and `vols`.

        """
        means.check_aligned(vols)

        values = stats.norm.ppf(q=1 - self._cf, loc=means.values, scale=vols.values)
        return TimeSeries(dates=means.dates, values=values)
