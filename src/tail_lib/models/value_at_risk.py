"""Value-at-risk models."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal
from typing import assert_never

import numpy as np

from tail_lib._checks import check_positive
from tail_lib._checks import check_probability
from tail_lib.distributions import Gaussian
from tail_lib.graph import Node
from tail_lib.graph import RunInput
from tail_lib.models.mean import MeanModel
from tail_lib.models.mean import ZeroMean
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tail_lib.windows import Windows

if TYPE_CHECKING:
    from numpy.lib._function_base_impl import (
        _InterpolationMethod as NumpyInterpolationMethod,
    )

    from tail_lib.models.volatility import FilteredReturns
    from tail_lib.models.volatility import VolatilityFilter
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


@dataclass(frozen=True)
class ValueAtRiskModel(Node[TimeSeries]):
    """Base class for value-at-risk models.

    A value-at-risk model is a node whose output is the value-at-risk on each
    date. Subclasses implement `compute`, from the returns and the outputs of
    their inputs, see `Node`.

    Building one raises `ValueError` if the confidence level isn't in (0, 1).

    Attributes:
        cf (float): The confidence level of the value-at-risk, keyword-only.

    """

    cf: float = field(kw_only=True)

    def __post_init__(self) -> None:
        check_probability("Confidence level", self.cf)
        super().__post_init__()


@dataclass(frozen=True)
class ValueAtRiskInput(RunInput, ValueAtRiskModel):
    """Value-at-risk forecasts passed in at run time, under a name (R13).

    The value-at-risk on date `t` is the input's value on date `t`. Like every
    value-at-risk model's, it should be the forecast for date `t + 1`, using
    data up to and including `t`, at confidence level `cf`: the library cannot
    check this. Backtests read `cf` from here. The cache reads it from the run
    inputs, see `RunInput`.

    Building it raises `ValueError` if the name isn't a Python identifier, or
    the confidence level isn't in (0, 1).

    Attributes:
        name (str): The name of the input, as passed to `run`: a Python
            identifier.
        cf (float): The confidence level the forecasts were computed at.

    """


@dataclass(frozen=True, kw_only=True)
class HistoricalSimulationsModel(ValueAtRiskModel):
    """Value-at-risk as an empirical quantile of the returns in a trailing window.

    The value-at-risk on the last date `t` of a window is the `1 - cf`
    empirical quantile of the returns in that window, i.e. of the `length`
    returns up to and including `t`:

        VaR[t] = Q_{1 - cf}(r[t - length + 1], ..., r[t])

    It is the forecast for the return of date `t + 1`. With a `step` above 1,
    each value-at-risk is held until the next window ends. Dates before the
    first full window are `NaN`. With a `Last` window, there is one
    value-at-risk, on the last date.

    The quantile interpolates between order statistics as set by
    `interpolation`: "linear" (numpy's default), "below" (the order statistic
    at or below the quantile) or "above" (the one at or above it).

    Reference: Jorion, *Value at Risk: The New Benchmark for Managing Financial
    Risk*, 3rd edition, McGraw-Hill, 2007.

    Attributes:
        cf (float): The confidence level, in (0, 1).
        window (Trailing | Last): The windows the quantiles are taken on.
        interpolation (InterpolationMethod): How the quantile interpolates
            between order statistics.

    """

    window: Trailing | Last
    interpolation: InterpolationMethod = "linear"

    def compute(self, returns: TimeSeries, /) -> TimeSeries:
        """Compute the value-at-risk on each window.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            TimeSeries: The value-at-risk, on the dates of the returns.

        """
        method = _interp_to_numpy_quantile_method(self.interpolation)
        return Windows(returns, self.window).apply(
            lambda w: np.quantile(w, q=1 - self.cf, method=method, axis=0)
        )


@dataclass(frozen=True, kw_only=True)
class FilteredHistoricalSimulationsModel(ValueAtRiskModel):
    """Historical simulation on returns standardized by their mean and volatility.

    The returns are standardized by `filter`, a `VolatilityFilter`, with the
    mean and volatility forecast for their date:

        z[s] = (r[s] - mean[s - 1]) / vol[s - 1]

    The value-at-risk on the last date `t` of a window is the `1 - cf`
    empirical quantile of the `length` standardized returns up to and
    including `t`, rescaled with the filter's mean and volatility on date `t`:

        VaR[t] = mean[t] + vol[t] * Q_{1 - cf}(z[t - length + 1], ..., z[t])

    It is the forecast for the return of date `t + 1`, using data up to and
    including `t`, like its inputs. The windows start on the first date where
    every series has a standardized return, after the warm-up of the
    volatility (R16): with an EWMA volatility backcast on 20 dates, the first
    window of 500 dates holds the standardized returns of dates 20 to 519.
    With a `step` above 1, each value-at-risk is held until the next window
    ends. Dates before the first full window are `NaN`. With a `Last` window,
    there is one value-at-risk, on the last date. `interpolation` is as in
    `HistoricalSimulationsModel`.

    Share one filter between the models that standardize the same way, e.g.
    on windows of 500 and 750 dates, so it is computed once.

    References: Hull and White, "Incorporating volatility updating into the
    historical simulation method for value-at-risk", Journal of Risk, 1998;
    Barone-Adesi, Giannopoulos and Vosper, "VaR without correlations for
    portfolios of derivative securities", Journal of Futures Markets, 1999.

    Attributes:
        cf (float): The confidence level, in (0, 1).
        filter (VolatilityFilter): Standardizes the returns. Its mean and
            volatility also rescale the quantile.
        window (Trailing | Last): The windows of standardized returns.
        interpolation (InterpolationMethod): How the quantile interpolates
            between order statistics.

    """

    filter: VolatilityFilter
    window: Trailing | Last
    interpolation: InterpolationMethod = "linear"

    # The field is named after the spec's `filter`, so its keyword is too.
    def compute(
        self,
        returns: TimeSeries,
        /,
        *,
        filter: FilteredReturns,  # noqa: A002
    ) -> TimeSeries:
        """Compute the value-at-risk on each window.

        Args:
            returns (TimeSeries): The returns. Only their dates and shape are
                used, to check the filter's output.
            filter (FilteredReturns): The output of the `filter` node: the
                standardized returns, with the means and volatilities that
                rescale the quantiles.

        Returns:
            TimeSeries: The value-at-risk, on the dates of the returns.

        Raises:
            ValueError: If the filter's output does not match the returns.

        """
        returns.check_aligned(filter.standardized)

        windows = Windows(filter.standardized, self.window, skip_leading_nan=True)
        method = _interp_to_numpy_quantile_method(self.interpolation)
        quantiles = np.quantile(windows.values, q=1 - self.cf, method=method, axis=0)
        # Each window is rescaled with the mean and volatility of its last date.
        ends = windows.end_indices
        mean, vol = filter.mean.values[ends], filter.vol.values[ends]
        return windows.to_series(mean + vol * quantiles)


@dataclass(frozen=True, kw_only=True)
class GaussianModel(ValueAtRiskModel):
    """Value-at-risk of normally distributed returns.

    The value-at-risk on date `t` is the `1 - cf` quantile of a normal
    distribution with the mean and volatility on date `t`:

        VaR[t] = mean[t] + vol[t] * Phi^-1(1 - cf)

    with `Phi^-1` the standard normal quantile, `Gaussian().quantile`. The
    volatility is the scale of the distribution: it has to be positive, or
    `NaN` where it is unknown (e.g. during a warm-up), which gives a `NaN`
    value-at-risk.

    It is the forecast for the return of date `t + 1`, using data up to and
    including `t`, like its inputs.

    Attributes:
        cf (float): The confidence level, in (0, 1).
        vol (VolatilityModel): The volatility, the scale of the
            distribution.
        mean (MeanModel): The mean of the distribution. Defaults to
            zero.

    """

    vol: VolatilityModel
    mean: MeanModel = field(default_factory=ZeroMean)

    def compute(
        self, returns: TimeSeries, /, *, vol: TimeSeries, mean: TimeSeries
    ) -> TimeSeries:
        """Compute the value-at-risk on each date.

        Args:
            returns (TimeSeries): The returns. Only their dates and shape are
                used, to check the inputs.
            vol (TimeSeries): The volatilities of the returns, the output of
                the `vol` node.
            mean (TimeSeries): The means of the returns, the output of the
                `mean` node.

        Returns:
            TimeSeries: The value-at-risk, on the dates of the returns.

        Raises:
            ValueError: If the means or volatilities don't match the returns,
                or if a volatility is zero or negative.

        """
        returns.check_aligned(vol)
        returns.check_aligned(mean)
        check_positive("Volatility", vol.values)

        values = mean.values + vol.values * Gaussian().quantile(1 - self.cf)
        return TimeSeries(dates=returns.dates, values=values)
