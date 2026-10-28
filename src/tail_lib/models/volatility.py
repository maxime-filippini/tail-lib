"""Models of the volatility of the returns."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from tail_lib.graph import Node
from tail_lib.graph import Preprocessor
from tail_lib.graph import RunInput
from tail_lib.graph import frozen_floats
from tail_lib.models.ewma import DEFAULT_EWMA_START
from tail_lib.models.ewma import DecayFactors
from tail_lib.models.ewma import DecayModel
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import EwmaStart
from tail_lib.models.ewma import ewma_variance
from tail_lib.models.mean import MeanModel
from tail_lib.models.mean import ZeroMean
from tail_lib.time_series.time_series import TimeSeries

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray
    from tail_lib._typing import Floats


class VolatilityModel(Preprocessor):
    """A model of the volatility of the returns, as of each date.

    The volatility on date `t` is the forecast of the volatility of the return
    of date `t + 1`, using data up to and including `t`.
    """


@dataclass(frozen=True)
class VolatilityInput(RunInput, VolatilityModel):
    """Volatilities passed in at run time, under a name (R13).

    The volatility on date `t` is the input's value on date `t`. Like every
    volatility model's, it should be the forecast for date `t + 1`, using data
    up to and including `t`: the library cannot check this. The cache reads
    it from the run inputs, see `RunInput`.

    Building it raises `ValueError` if the name isn't a Python identifier.

    Attributes:
        name (str): The name of the input, as passed to `run`: a Python
            identifier.

    """


@dataclass(frozen=True)
class ConstantVolatility(VolatilityModel):
    """The same volatility on every date.

    The volatility on date `t` is `vol`, for every `t`. It uses no returns.

    Attributes:
        vol (FloatArray | Floats | float): The volatility, a single value or
            an array broadcastable to the batch shape of the returns. An
            array is stored as nested tuples, see `frozen_floats`.

    """

    vol: FloatArray | Floats | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "vol", frozen_floats(self.vol))
        super().__post_init__()

    def compute(self, returns: TimeSeries, /) -> TimeSeries:
        """Repeat the volatility on every date of the returns.

        Args:
            returns (TimeSeries): The returns time series. Only its dates and
                shape are used.

        Returns:
            TimeSeries: The volatility, with the dates and shape of `returns`.

        """
        return TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(np.asarray(self.vol), returns.values.shape).copy(),
        )


@dataclass(frozen=True)
class EwmaVolatility(VolatilityModel):
    """Exponentially weighted moving average (EWMA) volatility.

    With `r[t]` the return on date `t` and `d` a decay factor, the variance
    follows the RiskMetrics recursion (J.P. Morgan/Reuters, RiskMetrics
    Technical Document, 1996, section 5.2.1):

        var[t] = d * var[t - 1] + (1 - d) * r[t] ** 2

    The volatility on date `t` is `sqrt(var[t])`: the forecast for date
    `t + 1`, using returns up to and including `t`. Each run of the recursion
    gets its first variance from `start`: a backcast by default, see
    `Backcast` and `StartingVol`. Dates before it are `NaN`.

    The decay model sets the runs, one per refit of its decay factor (see
    `DecayFactors`). Each runs with the decay of its refit, from the first
    date of the returns that decay was fitted on, and gives the volatilities
    from its refit date up to the date before the next refit:

    - a constant decay (`ConstantDecay`) has one refit, on the first date, so
      the recursion runs once, over the whole series;
    - a calibrated decay (`EwmaDecayCalibration`) is re-filtered over each
      training window, then continued until the next calibration (D7). So
      the volatility on a date only uses the returns of its training window
      and of later dates before it, and is `NaN` before the first window
      ends. With a `Last` window, there is one volatility, on the last date.

    This is how GARCH-type models are forecast after a refit, e.g. `rugarch`'s
    `ugarchroll` with a moving window, and `arch`'s `forecast` after
    `fit(first_obs=..., last_obs=...)`.

    A calibration re-filtered from another start than the one it was fitted
    with would forecast with a recursion it wasn't fitted on, so building an
    EWMA volatility raises `ValueError` if `start` differs from its
    calibration's.

    Attributes:
        decay (DecayModel): The decay factor, constant or calibrated.
        start (EwmaStart): How each run of the recursion gets its first
            variance. With a calibration, the same as the calibration's.

    """

    decay: DecayModel
    start: EwmaStart = field(default=DEFAULT_EWMA_START, kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.decay, EwmaDecayCalibration) and (
            self.decay.start != self.start
        ):
            msg = (
                f"The EWMA starts with {self.start!r}, but its calibration was "
                f"fitted starting with {self.decay.start!r}: give both the same "
                f"start!"
            )
            raise ValueError(msg)
        super().__post_init__()

    def compute(self, returns: TimeSeries, /, *, decay: DecayFactors) -> TimeSeries:
        """Compute the volatilities, with one run of the recursion per refit.

        Args:
            returns (TimeSeries): The returns time series.
            decay (DecayFactors): The decay factor on each date, with the dates
                and shape of `returns`, and its refits. Run `k` uses the decay
                of refit `k` on every date it covers.

        Returns:
            TimeSeries: The volatilities, on the dates of `returns`.

        Raises:
            ValueError: If the decay factor does not match the returns.

        """
        returns.check_aligned(decay.values)
        r = returns.values
        n = returns.n_dates

        # Run k starts on the first date its decay was fitted on, and gives
        # the volatilities from its refit to the date before the next refit.
        refits = decay.refits
        stops = np.append(refits[1:], n)
        firsts = decay.fit_starts

        # Every run at once, the runs on a batch axis: the returns of run k
        # are those of dates `firsts[k]` to `stops[k] - 1`, padded with NaN to
        # the longest run. The padding comes after the dates a run gives.
        length = int(np.max(stops - firsts))
        dates = firsts[:, None] + np.arange(length)  # (n_runs, length)
        padded = np.concatenate([r, np.full((1, *r.shape[1:]), np.nan)])
        runs = padded[np.where(dates < stops[:, None], dates, n).T]
        d = np.broadcast_to(decay.values.values[refits], runs.shape)
        var = np.moveaxis(ewma_variance(runs, d, self.start), 0, 1)

        # Row j of run k is the variance of date `firsts[k] + j`: keep the
        # dates from its refit to the date before the next one.
        given = (dates >= refits[:, None]) & (dates < stops[:, None])
        vols = np.full(r.shape, np.nan)
        vols[dates[given]] = np.sqrt(var[given])
        return TimeSeries(dates=returns.dates, values=vols)


@dataclass(frozen=True, eq=False)
class FilteredReturns:
    """The output of a volatility filter: standardized returns, and how.

    A filtered historical simulation rescales its quantiles with the mean and
    volatility that standardized the returns. They come with the standardized
    returns, so the simulation reads its filter's output alone.

    Attributes:
        standardized (TimeSeries): The standardized returns, with the dates
            and shape of the returns: `z[t] = (r[t] - mean[t - 1]) / vol[t - 1]`.
        mean (TimeSeries): The means, the output of the filter's `mean`. The
            mean on date `t` is the forecast for `t + 1`.
        vol (TimeSeries): The volatilities, the output of the filter's `vol`.
            The volatility on date `t` is the forecast for `t + 1`.

    """

    standardized: TimeSeries
    mean: TimeSeries
    vol: TimeSeries

    def __post_init__(self) -> None:
        self.standardized.check_aligned(self.mean)
        self.standardized.check_aligned(self.vol)


@dataclass(frozen=True)
class VolatilityFilter(Node[FilteredReturns]):
    """Returns standardized by the mean and volatility forecast for their date.

    The return of date `t` is filtered with the mean and volatility on date
    `t - 1`, the forecasts for date `t`:

        z[t] = (r[t] - mean[t - 1]) / vol[t - 1]

    The mean is zero by default, so the returns are only divided by their
    volatility. The first date is `NaN`, as are the dates before the first
    volatility, e.g. during an EWMA backcast. The output, `FilteredReturns`,
    also holds the mean and volatility, for the filtered historical
    simulation to rescale with.

    Attributes:
        vol (VolatilityModel): The volatility the returns are divided
            by.
        mean (MeanModel): The mean taken off the returns. Defaults to
            zero.

    """

    vol: VolatilityModel
    mean: MeanModel = field(default_factory=ZeroMean, kw_only=True)

    def compute(
        self, returns: TimeSeries, /, *, vol: TimeSeries, mean: TimeSeries
    ) -> FilteredReturns:
        """Filter the returns by their mean and volatility.

        Args:
            returns (TimeSeries): The returns time series.
            vol (TimeSeries): The volatilities, the output of the `vol` node,
                with the dates and shape of `returns`. The volatility on date
                `t` is the forecast for `t + 1`.
            mean (TimeSeries): The means, the output of the `mean` node, with
                the dates and shape of `returns`. The mean on date `t` is the
                forecast for `t + 1`.

        Returns:
            FilteredReturns: The standardized returns, `NaN` on the first date
                and where the mean or volatility of the previous date is, with
                the mean and volatility.

        Raises:
            ValueError: If the mean or volatility does not match the returns.

        """
        returns.check_aligned(mean)
        returns.check_aligned(vol)
        values = np.full(returns.values.shape, np.nan)
        values[1:] = (returns.values[1:] - mean.values[:-1]) / vol.values[:-1]
        return FilteredReturns(
            standardized=TimeSeries(dates=returns.dates, values=values),
            mean=mean,
            vol=vol,
        )
