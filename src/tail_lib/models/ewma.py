"""The EWMA method: its starting point, recursion and decay factors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from typing import assert_never

import numpy as np

from tail_lib._checks import check_positive
from tail_lib._checks import check_unit_interval
from tail_lib.graph import Cache
from tail_lib.graph import Preprocessor
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tail_lib.windows import Windows

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray
    from tail_lib.graph import Dependencies


@dataclass(frozen=True)
class Backcast:
    """Start the EWMA from the mean squared return of the first dates.

    The variance as of date `window - 1`, the forecast for date `window`, is
    the mean of the squared returns of dates `0` to `window - 1`, assuming a
    zero mean. The recursion continues from there.
    """

    window: int = 20

    def __post_init__(self) -> None:
        if self.window < 1:
            msg = f"Backcast window has to be at least 1, got {self.window}!"
            raise ValueError(msg)


@dataclass(frozen=True)
class StartingVol:
    """Start the EWMA from a given volatility, the forecast for the first date.

    The first variance as of a date is then the update of `vol ** 2` with the
    return of date `0`, on date `0`.
    """

    vol: FloatArray | float

    def __post_init__(self) -> None:
        check_positive("Starting volatility", self.vol)


type EwmaStart = Backcast | StartingVol
"""How an EWMA variance recursion gets its first value."""

DEFAULT_EWMA_START = Backcast(window=20)

DEFAULT_DECAY_BOUNDS = (0.8, 0.9999)
"""The default search interval of a decay factor fit."""

DEFAULT_FIT_ITERATIONS = 40
"""The default number of golden-section iterations of a decay factor fit."""


def starting_variance(r: FloatArray, start: EwmaStart) -> tuple[int, FloatArray]:
    """Date the EWMA recursion starts on, and the variance on that date.

    Args:
        r (FloatArray): The returns, with time on the first axis.
        start (EwmaStart): How the recursion gets its first value.

    Returns:
        tuple[int, FloatArray]: The index of the first date with a variance,
            and that variance, with the batch shape of `r`.

    Raises:
        ValueError: If a backcast needs more dates than there are.

    """
    match start:
        case Backcast(window=window):
            if r.shape[0] <= window:
                msg = (
                    f"Backcast window of {window} dates needs more than "
                    f"{window} dates of returns, got {r.shape[0]}!"
                )
                raise ValueError(msg)
            return window, np.mean(r[:window] ** 2, axis=0)
        case StartingVol(vol=vol):
            return 0, np.broadcast_to(np.square(vol), r.shape[1:])
        case _:
            assert_never(start)


def ewma_update(var: FloatArray, d: FloatArray | float, r: FloatArray) -> FloatArray:
    """One step of the EWMA variance recursion.

    Updates the variance forecast with the return of one date:

        d * var + (1 - d) * r ** 2

    It is the only place the recursion is written: `ewma_variance` and
    `ewma_neg_log_likelihood` both step through it.

    Args:
        var (FloatArray): The variance forecast for the date of `r`.
        d (FloatArray | float): The decay factor.
        r (FloatArray): The return of one date.

    Returns:
        FloatArray: The variance forecast for the next date.

    """
    return d * var + (1 - d) * r**2


def ewma_variance(r: FloatArray, d: FloatArray, start: EwmaStart) -> FloatArray:
    """Run the EWMA variance recursion along the first axis.

    The variance on date `t` is the forecast for date `t + 1`, using returns up
    to and including `t`:

        var[t] = d[t] * var[t - 1] + (1 - d[t]) * r[t] ** 2

    Args:
        r (FloatArray): The returns, with time on the first axis.
        d (FloatArray): The decay factor on each date, with the shape of `r`.
        start (EwmaStart): How the recursion gets its first variance.

    Returns:
        FloatArray: The variances, with the shape of `r`. `NaN` before the
            first date with a variance: date `window - 1` for a `Backcast`,
            date `0` for a `StartingVol`.

    """
    # `first` is the date the starting variance is the forecast for.
    first, forecast = starting_variance(r, start)
    var = np.full(r.shape, np.nan)
    if first > 0:
        var[first - 1] = forecast
    for t in range(first, r.shape[0]):
        forecast = ewma_update(forecast, d[t], r[t])
        var[t] = forecast
    return var


class DecayModel(Preprocessor):
    """A model of the EWMA decay factor, as of each date."""


class ConstantDecay(DecayModel):
    """The same decay factor on every date."""

    def __init__(self, decay_factor: FloatArray | float) -> None:
        """Build the model.

        Args:
            decay_factor (FloatArray | float): The decay factor, a single value
                or an array broadcastable to the batch shape of the returns,
                e.g. the output of `fit_ewma_decay`.

        Raises:
            ValueError: If the decay factor is not in [0,1].

        """
        self._decay_factor = np.asarray(decay_factor)
        check_unit_interval("Decay factor", self._decay_factor)

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        return self.compute(cache.returns)

    def compute(self, returns: TimeSeries) -> TimeSeries:
        """Repeat the decay factor on every date of the returns.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            TimeSeries: The decay factor, with the dates and shape of `returns`.

        """
        return TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(self._decay_factor, returns.values.shape).copy(),
        )


def ewma_neg_log_likelihood(
    r: FloatArray,
    decay_factor: FloatArray | float,
    *,
    start: EwmaStart = DEFAULT_EWMA_START,
) -> FloatArray:
    """Gaussian negative log-likelihood of zero-mean EWMA variance forecasts.

    The variance forecasts follow the recursion of `EwmaVolatility`, from the
    first date with a variance given by `start`. Dates before it are left out
    of the likelihood.

    Args:
        r (FloatArray): The returns, with time on the first axis.
        decay_factor (FloatArray | float): The decay factor, broadcastable to
            the batch shape of `r`.
        start (EwmaStart): How the recursion gets its first variance.

    Returns:
        FloatArray: The negative log-likelihood, one value per batch element
            (up to an additive constant).

    """
    first, var = starting_variance(r, start)
    nll = np.zeros(np.broadcast_shapes(r.shape[1:], np.shape(decay_factor)))
    for t in range(first, r.shape[0]):
        nll += np.log(var) + r[t] ** 2 / var
        var = ewma_update(var, decay_factor, r[t])

    return 0.5 * nll


def fit_ewma_decay(
    r: FloatArray,
    *,
    bounds: tuple[float, float] = DEFAULT_DECAY_BOUNDS,
    start: EwmaStart = DEFAULT_EWMA_START,
    n_iter: int = DEFAULT_FIT_ITERATIONS,
) -> FloatArray:
    """Fit EWMA decay factors by maximum likelihood.

    The fit is done along the first (time) axis, independently for every batch
    element, with a golden-section search vectorized over the batch.

    Args:
        r (FloatArray): The returns, with time on the first axis.
        bounds (tuple[float, float]): The search interval for the decay factor.
        start (EwmaStart): How the variance recursion gets its first value, see
            `ewma_neg_log_likelihood`.
        n_iter (int): The number of golden-section iterations. The search
            interval shrinks by a factor of about 0.618 per iteration.

    Returns:
        FloatArray: The decay factors, with the batch shape of `r`.

    Raises:
        ValueError: If the bounds are not in [0,1], or if a backcast needs
            more dates than there are.

    """
    lo, hi = bounds
    if not 0 <= lo < hi <= 1:
        msg = "Decay factor bounds have to satisfy 0 <= lower < upper <= 1!"
        raise ValueError(msg)

    # Fail early, before the search, if the start does not fit the returns.
    starting_variance(r, start)

    inv_phi = (np.sqrt(5) - 1) / 2
    a = np.full(r.shape[1:], lo)
    b = np.full(r.shape[1:], hi)
    for _ in range(n_iter):
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        nll_c = ewma_neg_log_likelihood(r, c, start=start)
        nll_d = ewma_neg_log_likelihood(r, d, start=start)
        is_left = nll_c < nll_d
        a, b = np.where(is_left, a, c), np.where(is_left, d, b)

    return (a + b) / 2


class EwmaDecayCalibration(DecayModel):
    """EWMA decay factors calibrated by maximum likelihood on trailing windows.

    Each window of returns, see `Windows`, is fitted with `fit_ewma_decay`. The
    decay factor on a date is the fit of the last window ending on or before
    it, so it only uses returns up to and including that date, and is `NaN`
    before the first window ends. With a `Last` window, there is one fit, on
    the last date. Share one instance between the models that
    should use the same calibration, so it is only fitted once.
    """

    def __init__(
        self,
        window: Trailing | Last,
        *,
        bounds: tuple[float, float] = DEFAULT_DECAY_BOUNDS,
        start: EwmaStart = DEFAULT_EWMA_START,
        n_iter: int = DEFAULT_FIT_ITERATIONS,
    ) -> None:
        """Build the calibration.

        Args:
            window (Trailing | Last): The training windows: how many dates
                each fit uses, and how many dates apart the fits are. `Last`
                fits once, on the last dates.
            bounds (tuple[float, float]): The search interval for the decay.
            start (EwmaStart): How the variance recursion of each fit gets its
                first value. A backcast uses the first dates of each window.
            n_iter (int): The number of golden-section iterations of each fit.

        """
        self._window = window
        self._bounds = bounds
        self._start = start
        self._n_iter = n_iter

    @property
    def window(self) -> Trailing | Last:
        """The training windows of the fits."""
        return self._window

    @property
    def start(self) -> EwmaStart:
        """How the variance recursion of each fit gets its first value."""
        return self._start

    def dependencies(self) -> Dependencies:  # noqa: D102
        return {}

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        return self.compute(cache.returns)

    def compute(self, returns: TimeSeries) -> TimeSeries:
        """Fit the decay factor on every training window.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            TimeSeries: The decay factor as of each date of `returns`.

        """
        fit = partial(
            fit_ewma_decay,
            bounds=self._bounds,
            start=self._start,
            n_iter=self._n_iter,
        )
        return Windows(returns, self._window).apply(fit)
