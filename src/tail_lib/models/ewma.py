"""The EWMA method: its starting point, recursion and decay factors."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING
from typing import assert_never

import numpy as np

from tail_lib._checks import check_positive
from tail_lib._checks import check_unit_interval
from tail_lib.graph import Node
from tail_lib.graph import frozen_floats
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import Trailing
from tail_lib.time_series.windows import Windows

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray
    from tail_lib._typing import Floats
    from tail_lib._typing import IntArray


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
    return of date `0`, on date `0`. An array of volatilities, one per
    series, is stored as nested tuples, see `frozen_floats`, so EWMA nodes
    starting from it can be identified by value.
    """

    vol: FloatArray | Floats | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "vol", frozen_floats(self.vol))
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


@dataclass(frozen=True, eq=False)
class DecayFactors:
    """The output of a decay model: the decay factor on each date, and its fits.

    A decay factor comes into force on a refit date, and stays in force until
    the next one. It was fitted on the returns from its fit's first date up to
    and including the refit date, so `EwmaVolatility` re-filters each one
    from that first date (D7). A constant decay has one refit, on the first
    date, fitted from it.

    Attributes:
        values (TimeSeries): The decay factor in force on each date, with the
            dates and shape of the returns. `NaN` before the first refit.
        refits (IntArray): The index of each refit date, in increasing order,
            shape (n_refits,). The same dates for every series.
        fit_starts (IntArray): For each refit, the index of the first date of
            the returns its fit used, shape (n_refits,).

    """

    values: TimeSeries
    refits: IntArray
    fit_starts: IntArray

    def __post_init__(self) -> None:
        n = self.values.n_dates
        if self.refits.ndim != 1 or self.refits.shape != self.fit_starts.shape:
            msg = (
                "Refits and fit starts have to be two arrays of one index per "
                f"refit, got shapes {self.refits.shape} and {self.fit_starts.shape}!"
            )
            raise ValueError(msg)
        if np.any(np.diff(self.refits) <= 0) or np.any(
            (self.refits < 0) | (self.refits >= n)
        ):
            msg = f"Refits have to be increasing dates of the {n} dates of values!"
            raise ValueError(msg)
        if np.any((self.fit_starts < 0) | (self.fit_starts > self.refits)):
            msg = "Each fit has to start on or before its refit date!"
            raise ValueError(msg)


class DecayModel(Node[DecayFactors]):
    """A model of the EWMA decay factor, as of each date, with its refits."""


@dataclass(frozen=True)
class ConstantDecay(DecayModel):
    """The same decay factor on every date.

    Building it raises `ValueError` if the decay factor is not in [0,1].

    Attributes:
        value (FloatArray | Floats | float): The decay factor, a single value
            or an array broadcastable to the batch shape of the returns, e.g.
            the output of `fit_ewma_decay`. An array is stored as nested
            tuples, see `frozen_floats`.

    """

    value: FloatArray | Floats | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", frozen_floats(self.value))
        check_unit_interval("Decay factor", self.value)
        super().__post_init__()

    def compute(self, returns: TimeSeries, /) -> DecayFactors:
        """Repeat the decay factor on every date of the returns.

        Args:
            returns (TimeSeries): The returns time series. Only its dates and
                shape are used.

        Returns:
            DecayFactors: The decay factor, with the dates and shape of
                `returns`, and one refit, on the first date, fitted from it.

        """
        values = TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(np.asarray(self.value), returns.values.shape).copy(),
        )
        first = np.zeros(1, dtype=np.int64)
        return DecayFactors(values=values, refits=first, fit_starts=first)


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


@dataclass(frozen=True)
class EwmaDecayCalibration(DecayModel):
    """EWMA decay factors calibrated by maximum likelihood on trailing windows.

    Each window of returns, see `Windows`, is fitted with `fit_ewma_decay`. The
    decay factor on a date is the fit of the last window ending on or before
    it, so it only uses returns up to and including that date, and is `NaN`
    before the first window ends. Each window's last date is a refit, and its
    first date the start of that fit (see `DecayFactors`). With a `Last`
    window, there is one fit, on the last date. Models given equal
    calibrations share one, so it is only fitted once (D9).

    Attributes:
        window (Trailing | Last): The training windows: how many dates each
            fit uses, and how many dates apart the fits are. `Last` fits
            once, on the last dates.
        bounds (tuple[float, float]): The search interval for the decay.
        start (EwmaStart): How the variance recursion of each fit gets its
            first value. A backcast uses the first dates of each window.
        n_iter (int): The number of golden-section iterations of each fit.

    """

    window: Trailing | Last
    bounds: tuple[float, float] = field(default=DEFAULT_DECAY_BOUNDS, kw_only=True)
    start: EwmaStart = field(default=DEFAULT_EWMA_START, kw_only=True)
    n_iter: int = field(default=DEFAULT_FIT_ITERATIONS, kw_only=True)

    def compute(self, returns: TimeSeries, /) -> DecayFactors:
        """Fit the decay factor on every training window.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            DecayFactors: The decay factor as of each date of `returns`, with a
                refit on the last date of each window, fitted from its first.

        """
        fit = partial(
            fit_ewma_decay,
            bounds=self.bounds,
            start=self.start,
            n_iter=self.n_iter,
        )
        windows = Windows(returns, self.window)
        ends = windows.end_indices
        return DecayFactors(
            values=windows.apply(fit),
            refits=ends,
            fit_starts=ends - (self.window.length - 1),
        )
