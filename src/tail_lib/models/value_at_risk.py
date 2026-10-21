"""Value-at-risk models."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import assert_never

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from tail_lib.time_series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Callable
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


class Preprocessor[**P](ABC):
    """A transformation of a returns time series into a model input.

    `compute` takes the returns, plus the outputs of the preprocessors this one
    depends on, as keyword arguments named after the keys of `preprocessors`.

    Preprocessors are identified by instance: within a model, or across the
    models of a `ModelPool`, each instance is only computed once.
    """

    # Declared as an attribute so subclasses can choose their own signature,
    # `compute(self, returns, **outputs_of_preprocessors)`.
    compute: Callable[..., TimeSeries]

    @property
    def preprocessors(self) -> Mapping[str, Preprocessor]:
        """Preprocessors keyed by the `compute` argument they feed."""
        return {}

    def run(self, returns: TimeSeries) -> TimeSeries:
        """Compute the preprocessors this one depends on, then this one.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            TimeSeries: The output of `compute`.

        """
        return _resolve(self, returns, cache={})


def _resolve(
    preprocessor: Preprocessor,
    returns: TimeSeries,
    cache: dict[Preprocessor, TimeSeries],
    pending: frozenset[Preprocessor] = frozenset(),
) -> TimeSeries:
    """Compute a preprocessor after its own preprocessors, reusing the cache."""
    if preprocessor in cache:
        return cache[preprocessor]

    if preprocessor in pending:
        msg = f"{type(preprocessor).__name__} depends on itself!"
        raise ValueError(msg)

    inputs = {
        name: _resolve(dependency, returns, cache, pending | {preprocessor})
        for name, dependency in preprocessor.preprocessors.items()
    }
    cache[preprocessor] = preprocessor.compute(returns, **inputs)
    return cache[preprocessor]


class Returns(Preprocessor):
    """Pass the returns through unchanged."""

    def compute(self, returns: TimeSeries) -> TimeSeries:  # noqa: D102
        return returns


class MeanModel(Preprocessor):
    """A model of the mean of the returns, as of each date."""


class ZeroMean(MeanModel):
    """A mean of zero on every date."""

    def compute(self, returns: TimeSeries) -> TimeSeries:  # noqa: D102
        return TimeSeries(
            dates=returns.dates,
            values=np.zeros(returns.values.shape),
        )


class ConstantMean(MeanModel):
    """The same mean on every date."""

    def __init__(self, mean: FloatArray | float) -> None:
        self._mean = np.asarray(mean)

    def compute(self, returns: TimeSeries) -> TimeSeries:  # noqa: D102
        return TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(self._mean, returns.values.shape).copy(),
        )


class VolatilityModel(Preprocessor):
    """A model of the volatility of the returns, as of each date."""


class ConstantVolatility(VolatilityModel):
    """The same volatility on every date."""

    def __init__(self, vol: FloatArray | float) -> None:
        self._vol = np.asarray(vol)

    def compute(self, returns: TimeSeries) -> TimeSeries:  # noqa: D102
        return TimeSeries(
            dates=returns.dates,
            values=np.broadcast_to(self._vol, returns.values.shape).copy(),
        )


class EwmaVolatility(VolatilityModel):
    """Exponentially weighted moving average volatility.

    The volatility on a date only uses the returns of the previous dates, the
    first one being `starting_vol`.

    `decay_factor` is either:
    - a single value, or an array broadcastable to the batch shape of the
      returns, for example the output of `fit_ewma_decay`;
    - a time series of decay factors on the dates of the returns, whose value on
      a date is used to update the volatility with the return of that date.
      A `NaN` decay factor gives `NaN` volatilities from the next date onwards.
    """

    def __init__(
        self,
        decay_factor: TimeSeries | FloatArray | float,
        starting_vol: FloatArray | float,
    ) -> None:
        self._decay_factor = (
            decay_factor
            if isinstance(decay_factor, TimeSeries)
            else np.asarray(decay_factor)
        )
        self._starting_vol = np.asarray(starting_vol)

        values = (
            decay_factor.values
            if isinstance(decay_factor, TimeSeries)
            else self._decay_factor
        )
        if np.any(values < 0) or np.any(values > 1):
            msg = "Decay factor has to be in [0,1]!"
            raise ValueError(msg)

    def compute(self, returns: TimeSeries) -> TimeSeries:  # noqa: D102
        r = returns.values
        decay_path = self._decay_path(returns)

        vols = np.zeros(r.shape)
        vols[0] = self._starting_vol
        for i in range(returns.n_dates - 1):
            df = decay_path[i]
            vols[i + 1] = np.sqrt(df * vols[i] ** 2 + (1 - df) * r[i] ** 2)

        return TimeSeries(dates=returns.dates, values=vols)

    def _decay_path(self, returns: TimeSeries) -> FloatArray:
        """Decay factors on every date, broadcast to the shape of the returns."""
        r = returns.values

        if isinstance(self._decay_factor, TimeSeries):
            if not np.array_equal(self._decay_factor.dates, returns.dates):
                msg = "Decay factor and returns have different dates!"
                raise ValueError(msg)
            df = self._decay_factor.values
            target_shape = r.shape
        else:
            df = self._decay_factor
            # A constant decay factor only broadcasts to the batch shape, so it
            # is never mistaken for a time-varying one.
            target_shape = r.shape[1:]

        msg = (
            f"Decay factor of shape {df.shape} does not broadcast to the "
            f"shape {target_shape}!"
        )
        try:
            broadcast_shape = np.broadcast_shapes(df.shape, target_shape)
        except ValueError as e:
            raise ValueError(msg) from e
        if broadcast_shape != target_shape:
            raise ValueError(msg)

        return np.broadcast_to(df, r.shape)


def ewma_neg_log_likelihood(
    r: FloatArray,
    decay_factor: FloatArray | float,
    *,
    burn_in: int,
) -> FloatArray:
    """Gaussian negative log-likelihood of zero-mean EWMA variance forecasts.

    The starting variance is the mean squared return over the first `burn_in`
    dates, which are then left out of the likelihood.

    Args:
        r (FloatArray): The returns, with time on the first axis.
        decay_factor (FloatArray | float): The decay factor, broadcastable to
            the batch shape of `r`.
        burn_in (int): The number of dates used for the starting variance.

    Returns:
        FloatArray: The negative log-likelihood, one value per batch element
            (up to an additive constant).

    """
    var = np.mean(r[:burn_in] ** 2, axis=0)
    nll = np.zeros(np.broadcast_shapes(r.shape[1:], np.shape(decay_factor)))
    for t in range(burn_in, r.shape[0]):
        nll += np.log(var) + r[t] ** 2 / var
        var = decay_factor * var + (1 - decay_factor) * r[t] ** 2

    return 0.5 * nll


def fit_ewma_decay(
    r: FloatArray,
    *,
    bounds: tuple[float, float] = (0.8, 0.9999),
    burn_in: int = 20,
    n_iter: int = 40,
) -> FloatArray:
    """Fit EWMA decay factors by maximum likelihood.

    The fit is done along the first (time) axis, independently for every batch
    element, with a golden-section search vectorized over the batch.

    Args:
        r (FloatArray): The returns, with time on the first axis.
        bounds (tuple[float, float]): The search interval for the decay factor.
        burn_in (int): The number of dates used for the starting variance, see
            `ewma_neg_log_likelihood`.
        n_iter (int): The number of golden-section iterations. The search
            interval shrinks by a factor of about 0.618 per iteration.

    Returns:
        FloatArray: The decay factors, with the batch shape of `r`.

    Raises:
        ValueError: If the bounds are not in [0,1], or if there are no dates
            left after the burn-in.

    """
    lo, hi = bounds
    if not 0 <= lo < hi <= 1:
        msg = "Decay factor bounds have to satisfy 0 <= lower < upper <= 1!"
        raise ValueError(msg)

    if burn_in < 1 or r.shape[0] <= burn_in:
        msg = (
            f"Burn-in has to be in [1, {r.shape[0]}) for {r.shape[0]} dates, "
            f"got {burn_in}!"
        )
        raise ValueError(msg)

    inv_phi = (np.sqrt(5) - 1) / 2
    a = np.full(r.shape[1:], lo)
    b = np.full(r.shape[1:], hi)
    for _ in range(n_iter):
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        nll_c = ewma_neg_log_likelihood(r, c, burn_in=burn_in)
        nll_d = ewma_neg_log_likelihood(r, d, burn_in=burn_in)
        is_left = nll_c < nll_d
        a, b = np.where(is_left, a, c), np.where(is_left, d, b)

    return (a + b) / 2


class CalibrationSchedule(ABC):
    """When, and on which returns, a model parameter is calibrated."""

    @abstractmethod
    def calibrate(
        self,
        returns: TimeSeries,
        fit: Callable[[FloatArray], FloatArray],
    ) -> TimeSeries:
        """Calibrate a parameter along the returns.

        Args:
            returns (TimeSeries): The returns time series.
            fit (Callable[[FloatArray], FloatArray]): Fits the parameter on an
                array of returns with time on the first axis, giving one value
                per batch element.

        Returns:
            TimeSeries: The parameter as of each date of the returns, only fitted
                on returns up to and including that date. `NaN` before the first
                calibration.

        """


class Rolling(CalibrationSchedule):
    """Calibration on a rolling window of returns.

    The first calibration uses the first `window` dates, then one happens every
    `refit_every` dates. Between calibrations, the last calibrated value holds.
    """

    def __init__(self, window: int, refit_every: int = 1) -> None:
        if window < 1:
            msg = "Window has to be at least 1!"
            raise ValueError(msg)

        if refit_every < 1:
            msg = "Refit frequency has to be at least 1!"
            raise ValueError(msg)

        self._window = window
        self._refit_every = refit_every

    def calibrate(  # noqa: D102
        self,
        returns: TimeSeries,
        fit: Callable[[FloatArray], FloatArray],
    ) -> TimeSeries:
        r = returns.values
        w = self._window
        if returns.n_dates < w:
            msg = f"Window of {w} dates is longer than the {returns.n_dates} dates!"
            raise ValueError(msg)

        # Training windows become a batch axis, right after the time axis.
        windows = np.moveaxis(sliding_window_view(r, w, axis=0), -1, 0)
        fitted = fit(windows[:, :: self._refit_every])

        # Map each date to the last calibration whose window ends on or before it.
        values = np.full((returns.n_dates, *fitted.shape[1:]), np.nan)
        fit_index = (np.arange(w - 1, returns.n_dates) - (w - 1)) // self._refit_every
        values[w - 1 :] = fitted[fit_index]

        return TimeSeries(dates=returns.dates, values=values)


class EwmaDecayCalibration(Preprocessor):
    """EWMA decay factors calibrated by maximum likelihood along a schedule.

    The output is the decay factor as of each date, see
    `CalibrationSchedule.calibrate`. Share one instance between the models
    that should use the same calibration, so it is only fitted once.
    """

    def __init__(
        self,
        schedule: CalibrationSchedule,
        *,
        bounds: tuple[float, float] = (0.8, 0.9999),
        burn_in: int = 20,
        n_iter: int = 40,
    ) -> None:
        self._schedule = schedule
        self._bounds = bounds
        self._burn_in = burn_in
        self._n_iter = n_iter

    @property
    def burn_in(self) -> int:
        """The number of dates used for the starting variance of each fit."""
        return self._burn_in

    def compute(self, returns: TimeSeries) -> TimeSeries:  # noqa: D102
        return self._schedule.calibrate(
            returns,
            fit=partial(
                fit_ewma_decay,
                bounds=self._bounds,
                burn_in=self._burn_in,
                n_iter=self._n_iter,
            ),
        )


class FittedEwmaVolatility(VolatilityModel):
    """EWMA volatility with decay factors from an `EwmaDecayCalibration`.

    The volatility on a date only uses decay factors calibrated on returns of
    previous dates. Volatilities are `NaN` up to and including the date of the
    first calibration.

    Before the first calibration, the volatility recursion runs with the first
    calibrated decay factor so that it is warmed up when volatilities start.
    This uses no returns beyond the first calibration window.
    """

    def __init__(
        self,
        calibration: EwmaDecayCalibration,
        *,
        starting_vol: FloatArray | float | None = None,
    ) -> None:
        self._calibration = calibration
        self._starting_vol = starting_vol

    @property
    def preprocessors(self) -> Mapping[str, Preprocessor]:  # noqa: D102
        return {"decay": self._calibration}

    def compute(self, returns: TimeSeries, *, decay: TimeSeries) -> TimeSeries:
        """Compute the volatilities.

        Args:
            returns (TimeSeries): The returns time series.
            decay (TimeSeries): The decay factor as of each date of `returns`.

        Returns:
            TimeSeries: The volatilities, on the dates of `returns`.

        Raises:
            ValueError: If no decay factor was calibrated.

        """
        is_calibrated = ~np.isnan(decay.values).reshape(decay.n_dates, -1).any(axis=1)
        if not is_calibrated.any():
            msg = "No calibration happened on the returns!"
            raise ValueError(msg)
        first = int(np.argmax(is_calibrated))

        warm_up_decay = decay.values.copy()
        warm_up_decay[:first] = decay.values[first]

        starting_vol = (
            np.sqrt(np.mean(returns.values[: self._calibration.burn_in] ** 2, axis=0))
            if self._starting_vol is None
            else self._starting_vol
        )

        vols = EwmaVolatility(
            decay_factor=TimeSeries(dates=decay.dates, values=warm_up_decay),
            starting_vol=starting_vol,
        ).compute(returns)

        values = vols.values
        values[: first + 1] = np.nan
        return TimeSeries(dates=returns.dates, values=values)


class VolatilityFilter(Preprocessor):
    """Returns divided by their volatility on the same date."""

    def __init__(self, vol_model: VolatilityModel) -> None:
        self._vol_model = vol_model

    @property
    def preprocessors(self) -> Mapping[str, Preprocessor]:  # noqa: D102
        return {"vols": self._vol_model}

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

    Each model declares its inputs as preprocessors, keyed by the name of the
    `compute` argument they feed. `run` evaluates the preprocessors on the
    returns, then calls `compute` with their outputs. `compute` signatures are
    free to differ across models.
    """

    _cf: float

    # Declared as an attribute so subclasses can choose their own signature.
    compute: Callable[..., T]

    def __init__(self, cf: float) -> None:
        if cf <= 0 or cf >= 1:
            msg = "Confidence level has to be in (0,1)!"
            raise ValueError(msg)

        self._cf = cf

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if "compute" not in cls.__dict__:
            msg = f"{cls.__name__} must define a `compute` method"
            raise TypeError(msg)

    @property
    @abstractmethod
    def preprocessors(self) -> Mapping[str, Preprocessor]:
        """Preprocessors keyed by the `compute` argument they feed."""

    def run(self, returns: TimeSeries) -> T:
        """Preprocess the returns, then compute the value-at-risk.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            T: The output of `compute`.

        """
        return self._run(returns, cache={})

    def _run(
        self,
        returns: TimeSeries,
        cache: dict[Preprocessor, TimeSeries],
    ) -> T:
        inputs = {
            name: _resolve(preprocessor, returns, cache)
            for name, preprocessor in self.preprocessors.items()
        }
        return self.compute(**inputs)


class ModelPool:
    """A set of models run on the same returns.

    Each distinct preprocessor instance is only computed once across the pool.
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
        cache: dict[Preprocessor, TimeSeries] = {}
        return {
            name: model._run(returns, cache=cache)  # noqa: SLF001
            for name, model in self._models.items()
        }


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
        self._returns = Returns()

    @property
    def preprocessors(self) -> Mapping[str, Preprocessor]:  # noqa: D102
        return {"returns": self._returns}

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

    @property
    def preprocessors(self) -> Mapping[str, Preprocessor]:  # noqa: D102
        return {"means": self._mean_model, "vols": self._vol_model}

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
