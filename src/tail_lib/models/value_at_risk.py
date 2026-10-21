"""Value-at-risk models."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import assert_never

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tail_lib.windows import Windows

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.lib._function_base_impl import (
        _InterpolationMethod as NumpyInterpolationMethod,
    )

    from tail_lib._typing import FloatArray


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


class Preprocessor(ABC):
    """A transformation of a returns time series into a model input.

    Subclasses implement `evaluate`, which gets the inputs of the preprocessor
    from a `Cache`, then computes its output. The computation itself usually
    lives in a `compute` method, whose arguments are the inputs.
    """

    @abstractmethod
    def evaluate(self, cache: Cache) -> TimeSeries:
        """Get the inputs from the cache, then compute the output.

        Args:
            cache (Cache): Holds the returns, and the outputs of other
                preprocessors on them.

        Returns:
            TimeSeries: The output of this preprocessor.

        """

    def run(self, returns: TimeSeries) -> TimeSeries:
        """Compute this preprocessor on the returns, with a fresh cache.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            TimeSeries: The output of this preprocessor.

        """
        return Cache(returns).get(self)


class Cache:
    """Outputs of preprocessors on one returns time series.

    Each preprocessor is computed at most once per cache. Preprocessors are
    identified by instance: share one instance between models to share its
    computation.
    """

    def __init__(self, returns: TimeSeries) -> None:
        """Build an empty cache for a returns time series.

        Args:
            returns (TimeSeries): The returns every preprocessor is computed on.

        """
        self._returns = returns
        self._outputs: dict[Preprocessor, TimeSeries] = {}
        self._pending: set[Preprocessor] = set()

    @property
    def returns(self) -> TimeSeries:
        """The returns every preprocessor is computed on."""
        return self._returns

    def get(self, preprocessor: Preprocessor) -> TimeSeries:
        """Get the output of a preprocessor, computing it on first request.

        Args:
            preprocessor (Preprocessor): The preprocessor.

        Returns:
            TimeSeries: The output of the preprocessor.

        Raises:
            ValueError: If the preprocessor needs its own output as an input.

        """
        if preprocessor in self._outputs:
            return self._outputs[preprocessor]

        if preprocessor in self._pending:
            msg = f"{type(preprocessor).__name__} depends on itself!"
            raise ValueError(msg)

        self._pending.add(preprocessor)
        try:
            output = preprocessor.evaluate(self)
        finally:
            self._pending.discard(preprocessor)

        self._outputs[preprocessor] = output
        return output


class MeanModel(Preprocessor):
    """A model of the mean of the returns, as of each date."""


class ZeroMean(MeanModel):
    """A mean of zero on every date.

    The mean on date `t` is `0`, for every series. It uses no returns.
    """

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


class VolatilityModel(Preprocessor):
    """A model of the volatility of the returns, as of each date."""


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


@dataclass(frozen=True)
class Backcast:
    """Start the EWMA from the mean squared return of the first dates.

    The variance on date `window` is the mean of the squared returns of dates
    `0` to `window - 1`, assuming a zero mean. The recursion starts on date
    `window`, so no date sees its own or later returns.
    """

    window: int = 20

    def __post_init__(self) -> None:
        if self.window < 1:
            msg = f"Backcast window has to be at least 1, got {self.window}!"
            raise ValueError(msg)


@dataclass(frozen=True)
class StartingVol:
    """Start the EWMA from a given volatility on the first date."""

    vol: FloatArray | float

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.vol) <= 0):
            msg = "Starting volatility has to be positive!"
            raise ValueError(msg)


type EwmaStart = Backcast | StartingVol
"""How an EWMA variance recursion gets its first value."""

DEFAULT_EWMA_START = Backcast(window=20)


def _starting_variance(r: FloatArray, start: EwmaStart) -> tuple[int, FloatArray]:
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
        if np.any(self._decay_factor < 0) or np.any(self._decay_factor > 1):
            msg = "Decay factor has to be in [0,1]!"
            raise ValueError(msg)

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


def _ewma_variance(r: FloatArray, d: FloatArray, start: EwmaStart) -> FloatArray:
    """Run the EWMA variance recursion along the first axis.

    Args:
        r (FloatArray): The returns, with time on the first axis.
        d (FloatArray): The decay factor on each date, with the shape of `r`.
        start (EwmaStart): How the recursion gets its first variance.

    Returns:
        FloatArray: The variances, with the shape of `r`. The variance on date
            `t` only uses returns of dates before `t`. `NaN` before the first
            date with a variance, see `_starting_variance`.

    """
    first, var0 = _starting_variance(r, start)
    var = np.full(r.shape, np.nan)
    var[first] = var0
    for t in range(first, r.shape[0] - 1):
        var[t + 1] = d[t] * var[t] + (1 - d[t]) * r[t] ** 2
    return var


class EwmaVolatility(VolatilityModel):
    """Exponentially weighted moving average (EWMA) volatility, with a fixed decay.

    With `r[t]` the return on date `t` and `d` the decay factor, the variance
    follows the RiskMetrics recursion (J.P. Morgan/Reuters, RiskMetrics
    Technical Document, 1996, section 5.2.1), over the whole series:

        var[t + 1] = d * var[t] + (1 - d) * r[t] ** 2

    The volatility on date `t` is `sqrt(var[t])`, and only uses returns of
    dates before `t`. The first variance comes from `start`: a backcast by
    default, see `Backcast` and `StartingVol`. Dates before it are `NaN`.

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

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        decay = cache.get(self._decay)
        return self.compute(cache.returns, decay=decay)

    def compute(self, returns: TimeSeries, *, decay: TimeSeries) -> TimeSeries:
        """Compute the volatilities.

        Args:
            returns (TimeSeries): The returns time series.
            decay (TimeSeries): The decay factor on each date, with the dates
                and shape of `returns`. The variance of date `t + 1` uses the
                decay factor of date `t`.

        Returns:
            TimeSeries: The volatilities, on the dates of `returns`.

        Raises:
            ValueError: If the decay factor does not match the returns.

        """
        returns.check_aligned(decay)
        var = _ewma_variance(returns.values, decay.values, self._start)
        return TimeSeries(dates=returns.dates, values=np.sqrt(var))


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
    first, var = _starting_variance(r, start)
    nll = np.zeros(np.broadcast_shapes(r.shape[1:], np.shape(decay_factor)))
    for t in range(first, r.shape[0]):
        nll += np.log(var) + r[t] ** 2 / var
        var = decay_factor * var + (1 - decay_factor) * r[t] ** 2

    return 0.5 * nll


def fit_ewma_decay(
    r: FloatArray,
    *,
    bounds: tuple[float, float] = (0.8, 0.9999),
    start: EwmaStart = DEFAULT_EWMA_START,
    n_iter: int = 40,
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
    _starting_variance(r, start)

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
    before the first window ends. Share one instance between the models that
    should use the same calibration, so it is only fitted once.
    """

    def __init__(
        self,
        window: Trailing,
        *,
        bounds: tuple[float, float] = (0.8, 0.9999),
        start: EwmaStart = DEFAULT_EWMA_START,
        n_iter: int = 40,
    ) -> None:
        """Build the calibration.

        Args:
            window (Trailing): The training windows: how many dates each fit
                uses, and how many dates apart the fits are.
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
    def window(self) -> Trailing:
        """The training windows of the fits."""
        return self._window

    @property
    def start(self) -> EwmaStart:
        """How the variance recursion of each fit gets its first value."""
        return self._start

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

    The volatility on date `t` only uses returns of dates before `t`. It is
    `NaN` up to and including the last date of the first training window.
    """

    def __init__(self, calibration: EwmaDecayCalibration) -> None:
        """Build the model.

        Args:
            calibration (EwmaDecayCalibration): The calibration of the decay
                factor. Its training windows and start are used to re-filter.

        """
        self._calibration = calibration

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
        window: Trailing,
        start: EwmaStart,
    ) -> TimeSeries:
        """Compute the volatilities.

        Args:
            returns (TimeSeries): The returns time series.
            decay (TimeSeries): The decay factor fitted on the training windows
                `window` of `returns`, as output by an `EwmaDecayCalibration`.
            window (Trailing): The training windows the decay was fitted on.
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
        length, step = window.length, window.step

        # Calibration k fits the training window of dates
        # [k * step, k * step + length - 1], and its decay factor is used for
        # the volatilities of the `step` dates after that window.
        window_ends = np.arange(length - 1, n, step)
        fitted = decay.values[window_ends]  # (n_fits, *batch)

        # Segment k: the training window, then the `step` dates after it. The
        # returns are padded with NaN so the last segments have full length;
        # the padding only reaches volatilities after the last date.
        segment_length = length + step
        padded = np.concatenate([r, np.full((segment_length, *r.shape[1:]), np.nan)])
        strided = sliding_window_view(padded, segment_length, axis=0)
        segments = np.moveaxis(strided, -1, 0)[:, window_ends - (length - 1)]

        # One recursion per segment, all at once: segments are a batch axis.
        d = np.broadcast_to(fitted, segments.shape)
        var = _ewma_variance(segments, d, start)

        # Row `length + j` of segment k is the variance of date
        # `window_ends[k] + 1 + j`, for j in [0, step).
        after_window = np.moveaxis(var[length:], 1, 0)  # (n_fits, step, *batch)
        flat = after_window.reshape(-1, *r.shape[1:])
        vols = np.full(r.shape, np.nan)
        n_after = min(flat.shape[0], n - length)
        vols[length : length + n_after] = np.sqrt(flat[:n_after])
        return TimeSeries(dates=returns.dates, values=vols)


class VolatilityFilter(Preprocessor):
    """Returns divided by their volatility on the same date."""

    def __init__(self, vol_model: VolatilityModel) -> None:
        self._vol_model = vol_model

    def evaluate(self, cache: Cache) -> TimeSeries:  # noqa: D102
        vols = cache.get(self._vol_model)
        return self.compute(cache.returns, vols=vols)

    def compute(self, returns: TimeSeries, *, vols: TimeSeries) -> TimeSeries:
        """Filter the returns by their volatility.

        Args:
            returns (TimeSeries): The returns time series.
            vols (TimeSeries): The volatilities, with the same dates and shape as
                `returns`.

        Returns:
            TimeSeries: The filtered returns, `NaN` where the volatility is.

        """
        returns.check_aligned(vols)
        return TimeSeries(dates=returns.dates, values=returns.values / vols.values)


class ValueAtRiskModel[T](ABC):
    """Base class for value-at-risk models.

    Subclasses implement `evaluate`, which gets the inputs of the model from a
    `Cache`, then computes the value-at-risk. The computation itself usually
    lives in a `compute` method, whose arguments are the inputs.
    """

    _cf: float

    def __init__(self, cf: float) -> None:
        if cf <= 0 or cf >= 1:
            msg = "Confidence level has to be in (0,1)!"
            raise ValueError(msg)

        self._cf = cf

    @abstractmethod
    def evaluate(self, cache: Cache) -> T:
        """Get the inputs from the cache, then compute the value-at-risk.

        Args:
            cache (Cache): Holds the returns, and the outputs of preprocessors
                on them.

        Returns:
            T: The value-at-risk.

        """

    def run(self, returns: TimeSeries) -> T:
        """Compute the value-at-risk on the returns, with a fresh cache.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            T: The value-at-risk.

        """
        return self.evaluate(Cache(returns))


class ModelPool:
    """A set of models run on the same returns.

    The models share one cache, so each preprocessor instance is only computed
    once across the pool.
    """

    def __init__(self, models: Mapping[str, ValueAtRiskModel[Any]]) -> None:
        self._models = dict(models)

    def run(self, returns: TimeSeries) -> dict[str, Any]:
        """Run every model of the pool on the returns.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            dict[str, Any]: The output of each model, keyed by model name.

        """
        cache = Cache(returns)
        return {name: model.evaluate(cache) for name, model in self._models.items()}


class HistoricalSimulationsModel(ValueAtRiskModel["FloatArray"]):
    """Value-at-risk as an empirical quantile of the returns."""

    _interpolation: InterpolationMethod

    def __init__(
        self,
        *,
        cf: float,
        interpolation: InterpolationMethod = "linear",
    ) -> None:
        super().__init__(cf=cf)
        self._interpolation = interpolation

    def evaluate(self, cache: Cache) -> FloatArray:  # noqa: D102
        return self.compute(cache.returns)

    def compute(self, returns: TimeSeries) -> FloatArray:
        """Compute the value-at-risk over the full returns sample.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            FloatArray: The value-at-risk, one value per series.

        """
        return np.quantile(
            returns.values,
            q=1 - self._cf,
            method=_interp_to_numpy_quantile_method(self._interpolation),
            axis=0,
        )


class GaussianModel(ValueAtRiskModel[TimeSeries]):
    """Value-at-risk of normally distributed returns."""

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
