"""Simulated returns, from an AR mean, an EWMA or GARCH volatility and shocks.

The returns of date `t` are its mean plus a residual, the volatility times a
shock:

    r[t] = mu[t] + eps[t],    eps[t] = sigma[t] * z[t]

The mean `mu[t]` comes from an `ArProcess`, the variance `sigma[t] ** 2` from
an `EwmaProcess` or a `GarchProcess`, both updated with the returns up to
`t - 1`. The shocks `z[t]` are independent draws from a `Distribution`, with
zero mean and unit variance, e.g. `Gaussian` or `StudentT`.

These are plain classes with set parameters, not nodes: a simulation reads no
returns. The recursions are the ones the volatility models use
(`ewma_update`, `garch_update`), so a model given the same parameters filters
the simulated returns back to the simulated volatilities.

Arrays follow the library's layout (R8): time on the first axis, then the
batch shape, e.g. one axis of paths. Every parameter is a single value or an
array broadcastable to the batch shape, e.g. one decay per series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tail_lib._shapes import check_broadcasts
from tail_lib.calendar import Calendar
from tail_lib.distributions import Distribution
from tail_lib.distributions import Gaussian
from tail_lib.models.ewma import ewma_update
from tail_lib.models.garch import garch_update
from tail_lib.time_series import TimeSeries

if TYPE_CHECKING:
    import datetime

    from tail_lib._typing import FloatArray


@dataclass(frozen=True)
class ArProcess:
    """An autoregressive (AR) mean of order `p`, the number of coefficients.

    The mean of the return of date `t` uses the returns of the `p` dates
    before it:

        mu[t] = const + ar[0] * r[t - 1] + ... + ar[p - 1] * r[t - p]

    Before the first date, the returns are at the long-run mean
    `const / (1 - sum(ar))`. With no coefficients (the default), the mean is
    `const` on every date. The process is only stationary if the roots of
    `1 - ar[0] x - ... - ar[p - 1] x ** p` lie outside the unit circle, which is
    not checked.

    Reference: Box and Jenkins, *Time Series Analysis: Forecasting and
    Control*, 1970.

    Args:
        const (FloatArray | float): The constant of the mean.
        ar (tuple[FloatArray | float, ...]): The coefficients, from lag 1 to
            lag `p`.

    """

    const: FloatArray | float = 0.0
    ar: tuple[FloatArray | float, ...] = ()

    def __post_init__(self) -> None:
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

    def forecast(self, past: list[FloatArray]) -> FloatArray:
        """Compute the mean of the next return.

        Args:
            past (list[FloatArray]): The last `order` returns, the latest
                first.

        Returns:
            FloatArray: The mean of the next return.

        """
        mean = np.asarray(self.const)
        for phi, r in zip(self.ar, past, strict=True):
            mean = mean + phi * r
        return mean

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("ArProcess", "const", self.const, batch_shape)
        for lag, phi in enumerate(self.ar, start=1):
            check_broadcasts("ArProcess", f"ar[lag {lag}]", phi, batch_shape)


@dataclass(frozen=True)
class EwmaProcess:
    """An EWMA variance, from a starting volatility.

    The variance of the return of date `t + 1` follows the RiskMetrics
    recursion (J.P. Morgan/Reuters, RiskMetrics Technical Document, 1996,
    section 5.2.1), on the residuals:

        sigma[t + 1] ** 2 = decay * sigma[t] ** 2 + (1 - decay) * eps[t] ** 2

    The variance of the first date is `start_vol ** 2`. The EWMA has no
    long-run variance, and simulated variances can drift to zero or explode
    on long paths: check the paths it gives.

    Args:
        decay (FloatArray | float): The decay factor, in [0, 1].
        start_vol (FloatArray | float): The volatility of the first date.

    """

    decay: FloatArray | float
    start_vol: FloatArray | float

    def __post_init__(self) -> None:
        decay = np.asarray(self.decay)
        if np.any(decay < 0) or np.any(decay > 1):
            msg = "Decay factor has to be in [0,1]!"
            raise ValueError(msg)
        if np.any(np.asarray(self.start_vol) <= 0):
            msg = "Starting volatility has to be positive!"
            raise ValueError(msg)

    def first_variance(self) -> FloatArray:
        """Give the variance of the first date, `start_vol ** 2`."""
        return np.square(self.start_vol)

    def update(self, var: FloatArray, eps: FloatArray) -> FloatArray:
        """Update the variance with a residual, see `ewma_update`."""
        return ewma_update(var, self.decay, eps)

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("EwmaProcess", "decay", self.decay, batch_shape)
        check_broadcasts("EwmaProcess", "start_vol", self.start_vol, batch_shape)


@dataclass(frozen=True)
class GarchProcess:
    """A GARCH(1, 1) variance, from its long-run value.

    The variance of the return of date `t + 1` follows the recursion of
    Bollerslev ("Generalized autoregressive conditional heteroskedasticity",
    Journal of Econometrics, 1986), on the residuals:

        sigma[t + 1] ** 2 = omega + alpha * eps[t] ** 2 + beta * sigma[t] ** 2

    The variance of the first date is the long-run variance
    `omega / (1 - alpha - beta)`, so `alpha + beta` has to be below 1.

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
        alpha = np.asarray(self.alpha)
        beta = np.asarray(self.beta)
        if np.any(np.asarray(self.omega) <= 0):
            msg = "GARCH omega has to be positive!"
            raise ValueError(msg)
        if np.any(alpha < 0) or np.any(beta < 0):
            msg = "GARCH alpha and beta have to be at least 0!"
            raise ValueError(msg)
        if np.any(alpha + beta >= 1):
            msg = "GARCH alpha + beta has to be below 1, for a long-run variance!"
            raise ValueError(msg)

    def long_run_variance(self) -> FloatArray:
        """Compute the long-run variance, `omega / (1 - alpha - beta)`."""
        return np.asarray(self.omega) / (1 - np.asarray(self.alpha) - self.beta)

    def first_variance(self) -> FloatArray:
        """Give the variance of the first date, the long-run variance."""
        return self.long_run_variance()

    def update(self, var: FloatArray, eps: FloatArray) -> FloatArray:
        """Update the variance with a residual, see `garch_update`."""
        return garch_update(
            var, eps, omega=self.omega, alpha=self.alpha, beta=self.beta
        )

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape."""
        check_broadcasts("GarchProcess", "omega", self.omega, batch_shape)
        check_broadcasts("GarchProcess", "alpha", self.alpha, batch_shape)
        check_broadcasts("GarchProcess", "beta", self.beta, batch_shape)


type VolatilityProcess = EwmaProcess | GarchProcess
"""How the variance of the simulated returns moves from date to date."""


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
    2. the variance of `t + 1` is updated with `eps[t]`, and the mean of
       `t + 1` with `r[t]`.

    Before the first date, the past returns are at the mean's long-run value,
    and the variance starts as the volatility process says. A burn-in
    simulates extra dates first and drops them, so the paths start away from
    these values.
    """

    def __init__(
        self,
        *,
        vol: VolatilityProcess,
        mean: ArProcess | None = None,
        shocks: Distribution | None = None,
    ) -> None:
        """Build the simulator.

        Args:
            vol (VolatilityProcess): The volatility of the returns.
            mean (ArProcess | None): The mean of the returns. Defaults to zero.
            shocks (Distribution | None): The distribution of the shocks.
                Defaults to `Gaussian`.

        """
        self._vol = vol
        self._mean = ArProcess() if mean is None else mean
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

        # The state before the first date: the past returns, the latest first,
        # and the mean and variance of the first date.
        long_run = np.broadcast_to(self._mean.long_run_mean(), batch_shape)
        past = [long_run] * self._mean.order
        mean = self._mean.forecast(past)
        var = np.broadcast_to(self._vol.first_variance(), batch_shape)

        for t in range(n):
            eps = np.sqrt(var) * z[t]
            returns[t] = mean + eps
            past = [returns[t], *past][: self._mean.order]
            var = self._vol.update(var, eps)
            mean = self._mean.forecast(past)
            # The forecasts for `t + 1`, stored on `t` (D1).
            vols[t] = np.sqrt(var)
            means[t] = mean

        dates = (Calendar() if calendar is None else calendar).dates_from(
            first_date, n_dates
        )
        return Simulation(
            returns=TimeSeries(dates=dates, values=returns[burn_in:]),
            means=TimeSeries(dates=dates, values=means[burn_in:]),
            vols=TimeSeries(dates=dates, values=vols[burn_in:]),
        )
