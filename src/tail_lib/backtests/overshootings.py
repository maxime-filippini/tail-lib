"""Overshootings of value-at-risk forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tail_lib.graph import Preprocessor
from tail_lib.time_series.time_series import TimeSeries

if TYPE_CHECKING:
    from tail_lib.models.value_at_risk import ValueAtRiskModel


def overshootings(var: TimeSeries, returns: TimeSeries) -> TimeSeries:
    """Flag the dates whose return falls below the previous date's VaR.

    The value-at-risk on date `t - 1` is the forecast for date `t`, and is a
    return quantile (a negative number for a loss). The return of date `t`
    overshoots it when

        r[t] < VaR[t - 1]

    A return equal to the value-at-risk is not an overshooting.

    Args:
        var (TimeSeries): The value-at-risk on each date.
        returns (TimeSeries): The returns, with the same dates and shape as
            `var`.

    Returns:
        TimeSeries: On the dates of the returns, 1.0 for an overshooting, 0.0
            for none, and `NaN` where there is nothing to compare: the first
            date, and dates whose return or previous value-at-risk is `NaN`.

    """
    returns.check_aligned(var)

    previous_var = var.values[:-1]
    r = returns.values[1:]
    comparable = ~np.isnan(previous_var) & ~np.isnan(r)

    values = np.full(returns.values.shape, np.nan)
    values[1:] = np.where(comparable, r < previous_var, np.nan)
    return TimeSeries(dates=returns.dates, values=values)


@dataclass(frozen=True)
class Overshootings(Preprocessor):
    """The overshootings of a value-at-risk model, as a node.

    Computes the model's value-at-risk, then flags its overshootings with
    `overshootings`. Backtests take this node rather than the model. The
    overshootings of equal models are one node, so the backtests of a
    model flag its overshootings once.

    Attributes:
        var (ValueAtRiskModel): The model whose forecasts are tested.

    """

    var: ValueAtRiskModel

    @property
    def cf_var(self) -> float:
        """The confidence level of the value-at-risk model."""
        return self.var.cf

    def compute(self, returns: TimeSeries, /, *, var: TimeSeries) -> TimeSeries:
        """Flag the overshootings, see `overshootings`.

        Args:
            returns (TimeSeries): The returns.
            var (TimeSeries): The value-at-risk, the output of the `var`
                node, on the dates of the returns.

        Returns:
            TimeSeries: The overshooting flags.

        """
        return overshootings(var, returns)
