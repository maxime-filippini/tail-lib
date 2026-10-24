"""Models of the volatility of the returns."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import assert_never

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tail_lib.graph import Cache
from tail_lib.graph import Preprocessor
from tail_lib.models.ewma import DEFAULT_EWMA_START
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import EwmaStart
from tail_lib.models.ewma import ewma_variance
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tail_lib.windows import Windows

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray
    from tail_lib.graph import Dependencies


class VolatilityModel(Preprocessor):
    """A model of the volatility of the returns, as of each date.

    The volatility on date `t` is the forecast of the volatility of the return
    of date `t + 1`, using data up to and including `t`.
    """


class VolatilityInput(VolatilityModel):
    """Volatilities passed in at run time, under a name (R13).

    The volatility on date `t` is the input's value on date `t`. Like every
    volatility model's, it should be the forecast for date `t + 1`, using data
    up to and including `t`: the library cannot check this.
    """

    def __init__(self, name: str) -> None:
        """Build the model.

        Args:
            name (str): The name of the input, as passed to `run`.

        """
        self._name = name

    @property
    def name(self) -> str:
        """The name of the input."""
        return self._name

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        return cache.input(self._name)


class ConstantVolatility(VolatilityModel):
    """The same volatility on every date.

    The volatility on date `t` is `vol`, for every `t`. It uses no returns.
    """

    def __init__(self, vol: FloatArray | float) -> None:
        """Build the model.

        Args:
            vol (FloatArray | float): The volatility, a single value or an
                array broadcastable to the batch shape of the returns.

        """
        self._vol = np.asarray(vol)

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        return self.compute(cache.returns)

    def compute(self, returns: TimeSeries) -> TimeSeries:
        """Repeat the volatility on every date of the returns.

        Args:
            returns (TimeSeries): The returns time series. Only its dates and
                shape are used.

        Returns:
            TimeSeries: The volatility, with the dates and shape of `returns`.

        """
        return TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(self._vol, returns.values.shape).copy(),
        )


class EwmaVolatility(VolatilityModel):
    """Exponentially weighted moving average (EWMA) volatility, with a fixed decay.

    With `r[t]` the return on date `t` and `d` the decay factor, the variance
    follows the RiskMetrics recursion (J.P. Morgan/Reuters, RiskMetrics
    Technical Document, 1996, section 5.2.1), over the whole series:

        var[t] = d * var[t - 1] + (1 - d) * r[t] ** 2

    The volatility on date `t` is `sqrt(var[t])`: the forecast for date
    `t + 1`, using returns up to and including `t`. The first variance comes
    from `start`: a backcast by default, see `Backcast` and `StartingVol`.
    Dates before it are `NaN`.

    For a decay recalibrated along the returns, see `FittedEwmaVolatility`.
    """

    def __init__(
        self,
        decay: ConstantDecay,
        *,
        start: EwmaStart = DEFAULT_EWMA_START,
    ) -> None:
        """Build the model.

        Args:
            decay (ConstantDecay): The decay factor.
            start (EwmaStart): How the recursion gets its first variance.

        """
        self._decay = decay
        self._start = start

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {"decay": self._decay}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        decay = cache.get(self._decay)
        return self.compute(cache.returns, decay=decay)

    def compute(self, returns: TimeSeries, *, decay: TimeSeries) -> TimeSeries:
        """Compute the volatilities.

        Args:
            returns (TimeSeries): The returns time series.
            decay (TimeSeries): The decay factor on each date, with the dates
                and shape of `returns`. The variance on date `t` uses the
                decay factor of date `t`.

        Returns:
            TimeSeries: The volatilities, on the dates of `returns`.

        Raises:
            ValueError: If the decay factor does not match the returns.

        """
        returns.check_aligned(decay)
        var = ewma_variance(returns.values, decay.values, self._start)
        return TimeSeries(dates=returns.dates, values=np.sqrt(var))


class FittedEwmaVolatility(VolatilityModel):
    """EWMA volatility, re-filtered with each calibration of the decay factor.

    Each calibration of `calibration` fits a decay factor on a training window
    of returns. With that decay factor, the EWMA recursion of `EwmaVolatility`
    is run over the training window, from its own start (`calibration.start`,
    e.g. a backcast on the window's first dates), then continued with the same
    decay factor over the dates until the next calibration. The volatility on
    those dates comes from that run: it uses a single decay factor, and only
    the returns of its training window and of later dates before it.

    This is how GARCH-type models are forecast after a refit, e.g. `rugarch`'s
    `ugarchroll` with a moving window, and `arch`'s `forecast` after
    `fit(first_obs=..., last_obs=...)`.

    Calibration `k` is used from the last date of its training window, the
    date it is known on, up to the date before the next calibration. The
    volatility on date `t` is the forecast for date `t + 1`, using returns up
    to and including `t`. It is `NaN` before the last date of the first
    training window. With a `Last` training window, there is one
    calibration, so one volatility, on the last date.
    """

    def __init__(self, calibration: EwmaDecayCalibration) -> None:
        """Build the model.

        Args:
            calibration (EwmaDecayCalibration): The calibration of the decay
                factor. Its training windows and start are used to re-filter.

        """
        self._calibration = calibration

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {"calibration": self._calibration}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        decay = cache.get(self._calibration)
        return self.compute(
            cache.returns,
            decay=decay,
            window=self._calibration.window,
            start=self._calibration.start,
        )

    def compute(
        self,
        returns: TimeSeries,
        *,
        decay: TimeSeries,
        window: Trailing | Last,
        start: EwmaStart,
    ) -> TimeSeries:
        """Compute the volatilities.

        Args:
            returns (TimeSeries): The returns time series.
            decay (TimeSeries): The decay factor fitted on the training windows
                `window` of `returns`, as output by an `EwmaDecayCalibration`.
            window (Trailing | Last): The training windows the decay was
                fitted on.
            start (EwmaStart): How the recursion gets its first variance, at
                the start of each training window.

        Returns:
            TimeSeries: The volatilities, on the dates of `returns`.

        Raises:
            ValueError: If the decay factor does not match the returns.

        """
        returns.check_aligned(decay)
        r = returns.values
        n = returns.n_dates
        length = window.length
        match window:
            case Trailing():
                step = window.step
            case Last():
                step = 1  # One calibration, used on the last date only.
            case _:
                assert_never(window)

        # Calibration k fits the training window ending on `window_ends[k]`,
        # the same windows as the calibration's, and its decay factor is used
        # for the volatilities of the `step` dates from the window's last date.
        window_ends = Windows(returns, window).end_indices
        fitted = decay.values[window_ends]  # (n_fits, *batch)

        # Segment k: the training window, then the `step - 1` dates after it.
        # The returns are padded with NaN so the last segments have full
        # length; the padding only reaches volatilities after the last date.
        segment_length = length + step - 1
        padded = np.concatenate([r, np.full((segment_length, *r.shape[1:]), np.nan)])
        strided = sliding_window_view(padded, segment_length, axis=0)
        segments = np.moveaxis(strided, -1, 0)[:, window_ends - (length - 1)]

        # One recursion per segment, all at once: segments are a batch axis.
        d = np.broadcast_to(fitted, segments.shape)
        var = ewma_variance(segments, d, start)

        # Row `length - 1 + j` of segment k is the variance of date
        # `window_ends[k] + j`, for j in [0, step). The segments follow each
        # other, so they fill the dates from the first window's last date.
        used = np.moveaxis(var[length - 1 :], 1, 0)  # (n_fits, step, *batch)
        flat = used.reshape(-1, *r.shape[1:])
        vols = np.full(r.shape, np.nan)
        first = window_ends[0]
        n_used = min(flat.shape[0], n - first)
        vols[first : first + n_used] = np.sqrt(flat[:n_used])
        return TimeSeries(dates=returns.dates, values=vols)


class VolatilityFilter(Preprocessor):
    """Returns divided by the volatility forecast for their date.

    The return of date `t` is divided by the volatility on date `t - 1`, the
    forecast for date `t`. The first date is `NaN`.
    """

    def __init__(self, vol_model: VolatilityModel) -> None:
        self._vol_model = vol_model

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {"vol_model": self._vol_model}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        vols = cache.get(self._vol_model)
        return self.compute(cache.returns, vols=vols)

    def compute(self, returns: TimeSeries, *, vols: TimeSeries) -> TimeSeries:
        """Filter the returns by their volatility.

        Args:
            returns (TimeSeries): The returns time series.
            vols (TimeSeries): The volatilities, with the same dates and shape as
                `returns`. The volatility on date `t` is the forecast for `t + 1`.

        Returns:
            TimeSeries: The filtered returns, `NaN` on the first date and where
                the volatility of the previous date is.

        """
        returns.check_aligned(vols)
        values = np.full(returns.values.shape, np.nan)
        values[1:] = returns.values[1:] / vols.values[:-1]
        return TimeSeries(dates=returns.dates, values=values)
