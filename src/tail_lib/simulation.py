"""Simulated returns, from a mean process, a volatility process and shocks.

The returns of date `t` are its mean plus a residual, the volatility times a
shock:

    r[t] = mu[t] + eps[t],    eps[t] = sigma[t] * z[t]

The mean `mu[t]` comes from a `MeanProcess` (`ConstantMeanProcess`,
`ArProcess`), the volatility `sigma[t]` from a `VolatilityProcess`
(`ConstantVolProcess`, `EwmaProcess`, `GarchProcess`), both updated with the
returns up to `t - 1`. The shocks `z[t]` are independent draws from a
`Distribution`, with zero mean and unit variance, e.g. `Gaussian` or
`StudentT`.

Each process owns its recursion. It holds its parameters, and its state is
an array passed in and out of its methods:

- `start(batch_shape)` gives the state before the first date;
- `value(state)` reads the mean (or volatility) of the next date from it;
- `update(state, x)` moves it forward one date, with the return `r[t]` for a
  mean process and the residual `eps[t]` for a volatility process.

The `Simulator` only chains these calls; it knows nothing of lags or
variances.

These are plain classes with set parameters, not nodes: a simulation reads no
returns. The recursions are the ones the volatility models use
(`ewma_update`, `garch_update`), so a model given the same parameters filters
the simulated returns back to the simulated volatilities.

Arrays follow the library's layout (R8): time on the first axis, then the
batch shape, e.g. one axis of paths. Every parameter is a single value or an
array broadcastable to the batch shape, e.g. one decay per series.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tail_lib._checks import check_array_positive
from tail_lib._checks import check_broadcasts
from tail_lib._checks import check_unit_interval
from tail_lib.calendar import Calendar
from tail_lib.distributions import Distribution
from tail_lib.distributions import Gaussian
from tail_lib.models.ewma import ewma_update
from tail_lib.models.garch import garch_update
from tail_lib.time_series.time_series import TimeSeries

if TYPE_CHECKING:
    import datetime

    from tail_lib._typing import FloatArray


class MeanProcess(ABC):
    """How the mean of the simulated returns moves from date to date.

    The state is an array, whose layout each process chooses. `value` reads
    the mean of the next return from it, and `update` adds one return.
    """

    @abstractmethod
    def start(self, batch_shape: tuple[int, ...]) -> FloatArray:
        """Give the state before the first date.

        Args:
            batch_shape (tuple[int, ...]): The shape of each date's returns.

        Returns:
            FloatArray: The state, whose `value` is the mean of the first
                date.

        """

    @abstractmethod
    def value(self, state: FloatArray) -> FloatArray:
        """Read the mean of the next return from the state.

        Args:
            state (FloatArray): The state, from `start` or `update`.

        Returns:
            FloatArray: The mean, with the batch shape.

        """

    @abstractmethod
    def update(self, state: FloatArray, r: FloatArray) -> FloatArray:
        """Move the state forward one date, with that date's return.

        Args:
            state (FloatArray): The state, whose `value` is the mean of `r`.
            r (FloatArray): The return of one date.

        Returns:
            FloatArray: The state, whose `value` is the mean of the next date.

        """

    @abstractmethod
    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""


class VolatilityProcess(ABC):
    """How the volatility of the simulated returns moves from date to date.

    The state is an array, whose layout each process chooses. `value` reads
    the volatility of the next return from it, and `update` adds one
    residual, the return minus its mean.
    """

    @abstractmethod
    def start(self, batch_shape: tuple[int, ...]) -> FloatArray:
        """Give the state before the first date.

        Args:
            batch_shape (tuple[int, ...]): The shape of each date's returns.

        Returns:
            FloatArray: The state, whose `value` is the volatility of the
                first date.

        """

    @abstractmethod
    def value(self, state: FloatArray) -> FloatArray:
        """Read the volatility of the next return from the state.

        Args:
            state (FloatArray): The state, from `start` or `update`.

        Returns:
            FloatArray: The volatility, with the batch shape.

        """

    @abstractmethod
    def update(self, state: FloatArray, eps: FloatArray) -> FloatArray:
        """Move the state forward one date, with that date's residual.

        Args:
            state (FloatArray): The state, whose `value` is the volatility of
                `eps`.
            eps (FloatArray): The residual of one date: its return minus its
                mean.

        Returns:
            FloatArray: The state, whose `value` is the volatility of the
                next date.

        """

    @abstractmethod
    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""


@dataclass(frozen=True)
class ConstantMeanProcess(MeanProcess):
    """The same mean on every date.

    The mean of the return of every date `t` is

        mu[t] = mean

    The state is the mean, broadcast to the batch shape, and never changes.

    Args:
        mean (FloatArray | float): The mean.

    """

    mean: FloatArray | float = 0.0

    def start(self, batch_shape: tuple[int, ...]) -> FloatArray:
        """Give the mean, broadcast to the batch shape."""
        return np.broadcast_to(np.asarray(self.mean, dtype=float), batch_shape)

    def value(self, state: FloatArray) -> FloatArray:
        """Give the mean: the state itself."""
        return state

    def update(self, state: FloatArray, r: FloatArray) -> FloatArray:  # noqa: ARG002
        """Keep the state: the mean doesn't depend on the returns."""
        return state

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("ConstantMeanProcess", "mean", self.mean, batch_shape)


@dataclass(frozen=True)
class ArProcess(MeanProcess):
    """An autoregressive (AR) mean of order `p`, the number of coefficients.

    The mean of the return of date `t` uses the returns of the `p` dates
    before it:

        mu[t] = const + ar[0] * r[t - 1] + ... + ar[p - 1] * r[t - p]

    The state is the last `p` returns, the latest first, stacked on a first
    axis of length `p`. Before the first date, they are at the long-run mean
    `const / (1 - sum(ar))`. The order is at least 1: for a mean that doesn't
    move, use `ConstantMeanProcess`. The process is only stationary if the
    roots of `1 - ar[0] x - ... - ar[p - 1] x ** p` lie outside the unit
    circle, which is not checked.

    Reference: Box and Jenkins, *Time Series Analysis: Forecasting and
    Control*, 1970.

    Args:
        ar (tuple[FloatArray | float, ...]): The coefficients, from lag 1 to
            lag `p`, at least one.
        const (FloatArray | float): The constant of the mean.

    """

    ar: tuple[FloatArray | float, ...]
    const: FloatArray | float = 0.0

    def __post_init__(self) -> None:
        if not self.ar:
            msg = (
                "ArProcess needs at least one AR coefficient: for a constant "
                "mean, use ConstantMeanProcess!"
            )
            raise ValueError(msg)
        if np.any(self._ar_sum() == 1):
            msg = "AR coefficients sum to 1: the mean has no long-run value!"
            raise ValueError(msg)

    @property
    def order(self) -> int:
        """The number of past returns the mean uses."""
        return len(self.ar)

    def _ar_sum(self) -> FloatArray:
        total = np.zeros(())
        for phi in self.ar:
            total = total + np.asarray(phi)
        return total

    def long_run_mean(self) -> FloatArray:
        """Compute the long-run mean, `const / (1 - sum(ar))`."""
        return np.asarray(self.const) / (1 - self._ar_sum())

    def start(self, batch_shape: tuple[int, ...]) -> FloatArray:
        """Give `order` past returns, all at the long-run mean."""
        long_run = np.broadcast_to(self.long_run_mean(), batch_shape)
        return np.stack([long_run] * self.order)

    def value(self, state: FloatArray) -> FloatArray:
        """Compute the mean, `const + sum(ar[k] * state[k])`."""
        mean = np.asarray(self.const)
        for phi, r in zip(self.ar, state, strict=True):
            mean = mean + phi * r
        return mean

    def update(self, state: FloatArray, r: FloatArray) -> FloatArray:
        """Put the return first and drop the oldest one."""
        return np.concatenate([r[np.newaxis], state[:-1]])

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("ArProcess", "const", self.const, batch_shape)
        for lag, phi in enumerate(self.ar, start=1):
            check_broadcasts("ArProcess", f"ar[lag {lag}]", phi, batch_shape)


@dataclass(frozen=True)
class ConstantVolProcess(VolatilityProcess):
    """The same volatility on every date.

    The volatility of the return of every date `t` is

        sigma[t] = vol

    The state is the volatility, broadcast to the batch shape, and never
    changes. With Gaussian shocks, the returns are i.i.d. Gaussian.

    Args:
        vol (FloatArray | float): The volatility, positive.

    """

    vol: FloatArray | float

    def __post_init__(self) -> None:
        check_array_positive("Volatility", self.vol)

    def start(self, batch_shape: tuple[int, ...]) -> FloatArray:
        """Give the volatility, broadcast to the batch shape."""
        return np.broadcast_to(np.asarray(self.vol, dtype=float), batch_shape)

    def value(self, state: FloatArray) -> FloatArray:
        """Give the volatility: the state itself."""
        return state

    def update(self, state: FloatArray, eps: FloatArray) -> FloatArray:  # noqa: ARG002
        """Keep the state: the volatility doesn't depend on the residuals."""
        return state

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("ConstantVolProcess", "vol", self.vol, batch_shape)


@dataclass(frozen=True)
class EwmaProcess(VolatilityProcess):
    """An EWMA variance, from a starting volatility.

    The variance of the return of date `t + 1` follows the RiskMetrics
    recursion (J.P. Morgan/Reuters, RiskMetrics Technical Document, 1996,
    section 5.2.1), on the residuals:

        sigma[t + 1] ** 2 = decay * sigma[t] ** 2 + (1 - decay) * eps[t] ** 2

    The state is the variance; the variance of the first date is
    `start_vol ** 2`. The EWMA has no long-run variance, and simulated
    variances can drift to zero or explode on long paths: check the paths it
    gives.

    Args:
        decay (FloatArray | float): The decay factor, in [0, 1].
        start_vol (FloatArray | float): The volatility of the first date.

    """

    decay: FloatArray | float
    start_vol: FloatArray | float

    def __post_init__(self) -> None:
        check_unit_interval("Decay factor", self.decay)
        check_array_positive("Starting volatility", self.start_vol)

    def start(self, batch_shape: tuple[int, ...]) -> FloatArray:
        """Give the variance of the first date, `start_vol ** 2`."""
        return np.broadcast_to(np.square(self.start_vol), batch_shape)

    def value(self, state: FloatArray) -> FloatArray:
        """Give the volatility, the square root of the variance."""
        return np.sqrt(state)

    def update(self, state: FloatArray, eps: FloatArray) -> FloatArray:
        """Update the variance with a residual, see `ewma_update`."""
        return ewma_update(state, self.decay, eps)

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("EwmaProcess", "decay", self.decay, batch_shape)
        check_broadcasts("EwmaProcess", "start_vol", self.start_vol, batch_shape)


@dataclass(frozen=True)
class GarchProcess(VolatilityProcess):
    """A GARCH(1, 1) variance, from its long-run value.

    The variance of the return of date `t + 1` follows the recursion of
    Bollerslev ("Generalized autoregressive conditional heteroskedasticity",
    Journal of Econometrics, 1986), on the residuals:

        sigma[t + 1] ** 2 = omega + alpha * eps[t] ** 2 + beta * sigma[t] ** 2

    The state is the variance; the variance of the first date is the
    long-run variance `omega / (1 - alpha - beta)`, so `alpha + beta` has to
    be below 1. To give the long-run variance instead of `omega`, see
    `from_long_run_variance`.

    Args:
        omega (FloatArray | float): The constant of the recursion, positive.
        alpha (FloatArray | float): The weight of the squared residual, at
            least 0.
        beta (FloatArray | float): The weight of the previous variance, at
            least 0.

    """

    omega: FloatArray | float
    alpha: FloatArray | float
    beta: FloatArray | float

    def __post_init__(self) -> None:
        # Alpha and beta first: with `alpha + beta >= 1`, an omega built by
        # `from_long_run_variance` isn't positive, and that is the error.
        alpha = np.asarray(self.alpha)
        beta = np.asarray(self.beta)
        if np.any(alpha < 0) or np.any(beta < 0):
            msg = "GARCH alpha and beta have to be at least 0!"
            raise ValueError(msg)
        if np.any(alpha + beta >= 1):
            msg = "GARCH alpha + beta has to be below 1, for a long-run variance!"
            raise ValueError(msg)
        check_array_positive("GARCH omega", self.omega)

    @classmethod
    def from_long_run_variance(
        cls,
        *,
        alpha: FloatArray | float,
        beta: FloatArray | float,
        long_run_variance: FloatArray | float,
    ) -> GarchProcess:
        """Build the process from its long-run variance instead of `omega`.

        The constant of the recursion is the one giving that long-run
        variance:

            omega = long_run_variance * (1 - alpha - beta)

        Args:
            alpha (FloatArray | float): The weight of the squared residual, at
                least 0.
            beta (FloatArray | float): The weight of the previous variance, at
                least 0, with `alpha + beta` below 1.
            long_run_variance (FloatArray | float): The long-run variance,
                positive, e.g. `0.01 ** 2` for a daily volatility of 1%.

        Returns:
            GarchProcess: The process, with `long_run_variance()` equal to
                `long_run_variance`.

        Raises:
            ValueError: If a parameter is outside its bounds.

        """
        check_array_positive("GARCH long-run variance", long_run_variance)
        omega = np.asarray(long_run_variance) * (1 - np.asarray(alpha) - beta)
        return cls(omega=omega, alpha=alpha, beta=beta)

    def long_run_variance(self) -> FloatArray:
        """Compute the long-run variance, `omega / (1 - alpha - beta)`."""
        return np.asarray(self.omega) / (1 - np.asarray(self.alpha) - self.beta)

    def start(self, batch_shape: tuple[int, ...]) -> FloatArray:
        """Give the variance of the first date, the long-run variance."""
        return np.broadcast_to(self.long_run_variance(), batch_shape)

    def value(self, state: FloatArray) -> FloatArray:
        """Give the volatility, the square root of the variance."""
        return np.sqrt(state)

    def update(self, state: FloatArray, eps: FloatArray) -> FloatArray:
        """Update the variance with a residual, see `garch_update`."""
        return garch_update(
            state, eps, omega=self.omega, alpha=self.alpha, beta=self.beta
        )

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("GarchProcess", "omega", self.omega, batch_shape)
        check_broadcasts("GarchProcess", "alpha", self.alpha, batch_shape)
        check_broadcasts("GarchProcess", "beta", self.beta, batch_shape)


@dataclass(frozen=True)
class Simulation:
    """Simulated returns, with their true means and volatilities.

    The means and volatilities follow the timing of the library's models
    (D1): the value on date `t` is the forecast for date `t + 1`, i.e.
    `mu[t + 1]` and `sigma[t + 1]`, known from the returns up to `t`. So they
    compare directly with a model's output on the same returns.
    """

    returns: TimeSeries
    means: TimeSeries
    vols: TimeSeries


class Simulator:
    """Simulates returns from a mean, a volatility and shocks.

    See the module docstring for the formula. On each date `t`, in order:

    1. the residual is `eps[t] = sigma[t] * z[t]`, and the return
       `r[t] = mu[t] + eps[t]`;
    2. the mean process is updated with `r[t]` and the volatility process
       with `eps[t]`, giving `mu[t + 1]` and `sigma[t + 1]`.

    The processes start as they say (`start`). A burn-in simulates extra
    dates first and drops them, so the paths start away from these values.
    """

    def __init__(
        self,
        *,
        vol: VolatilityProcess,
        mean: MeanProcess | None = None,
        shocks: Distribution | None = None,
    ) -> None:
        """Build the simulator.

        Args:
            vol (VolatilityProcess): The volatility of the returns.
            mean (MeanProcess | None): The mean of the returns. Defaults to
                a zero `ConstantMeanProcess`.
            shocks (Distribution | None): The distribution of the shocks.
                Defaults to `Gaussian`.

        """
        self._vol = vol
        self._mean = ConstantMeanProcess() if mean is None else mean
        self._shocks = Gaussian() if shocks is None else shocks

    # All arguments but `n_dates` are keyword-only, with defaults.
    def simulate(  # noqa: PLR0913
        self,
        n_dates: int,
        *,
        first_date: datetime.date,
        batch_shape: tuple[int, ...] = (),
        burn_in: int = 0,
        calendar: Calendar | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> Simulation:
        """Simulate the returns, their means and their volatilities.

        Args:
            n_dates (int): The number of dates to simulate, after the burn-in.
            first_date (datetime.date): The first date of the output.
            batch_shape (tuple[int, ...]): The shape of each date's returns,
                e.g. `(n_paths,)`. Every parameter has to broadcast to it.
            burn_in (int): The number of dates simulated, then dropped, before
                the first date.
            calendar (Calendar | None): The calendar of valid dates, see
                `TimeSeries.from_first_date`. Defaults to Monday to Friday.
            rng (np.random.Generator | int | None): The random generator, or a
                seed for one. The same seed gives the same paths.

        Returns:
            Simulation: The returns, means and volatilities, with shape
                `(n_dates, *batch_shape)`.

        Raises:
            ValueError: If `n_dates` is below 1, `burn_in` below 0, or a
                parameter doesn't broadcast to `batch_shape`.

        """
        if n_dates < 1:
            msg = f"Number of dates has to be at least 1, got {n_dates}!"
            raise ValueError(msg)
        if burn_in < 0:
            msg = f"Burn-in has to be at least 0, got {burn_in}!"
            raise ValueError(msg)
        self._mean.check_shape(batch_shape)
        self._vol.check_shape(batch_shape)
        self._shocks.check_shape(batch_shape)

        generator = np.random.default_rng(rng)
        n = burn_in + n_dates
        shape = (n, *batch_shape)
        z = self._shocks.draw(generator, shape)
        returns = np.empty(shape)
        means = np.empty(shape)
        vols = np.empty(shape)

        mean_state = self._mean.start(batch_shape)
        vol_state = self._vol.start(batch_shape)
        for t in range(n):
            eps = self._vol.value(vol_state) * z[t]
            returns[t] = self._mean.value(mean_state) + eps
            mean_state = self._mean.update(mean_state, returns[t])
            vol_state = self._vol.update(vol_state, eps)
            # The forecasts for `t + 1`, stored on `t` (D1).
            means[t] = self._mean.value(mean_state)
            vols[t] = self._vol.value(vol_state)

        dates = (Calendar() if calendar is None else calendar).dates_from(
            first_date, n_dates
        )
        return Simulation(
            returns=TimeSeries(dates=dates, values=returns[burn_in:]),
            means=TimeSeries(dates=dates, values=means[burn_in:]),
            vols=TimeSeries(dates=dates, values=vols[burn_in:]),
        )
