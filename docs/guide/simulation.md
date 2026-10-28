# Simulation

The [`Simulator`][tail_lib.simulation.Simulator] draws returns from a mean,
a volatility and shocks (R18). It also returns the true means and
volatilities, so you can check how well a model recovers them.

```python
import datetime

from tail_lib.distributions import StudentT
from tail_lib.simulation import ArProcess, GarchProcess, Simulator

sim = Simulator(
    mean=ArProcess(const=0.0, ar=(0.05,)),
    vol=GarchProcess(omega=2e-6, alpha=0.08, beta=0.90),
    shocks=StudentT(df=5),
).simulate(2520, first_date=datetime.date(2015, 1, 1), batch_shape=(100,), rng=42)

sim.returns  # TimeSeries of shape (2520, 100): 100 paths
sim.means  # the true mean, forecast for the next date
sim.vols  # the true volatility, forecast for the next date
```

## The process

On each date \(t\):

\[
\varepsilon_t = \sigma_t z_t, \qquad r_t = \mu_t + \varepsilon_t
\]

then the variance of \(t + 1\) is updated with \(\varepsilon_t\), and the mean
of \(t + 1\) with \(r_t\). `sim.means` and `sim.vols` hold, on date \(t\), the
mean and volatility of \(t + 1\): the [library's timing](concepts.md#timing-the-value-on-t-is-the-forecast-for-t-1),
so they compare directly with a model's output.

### Mean: `ConstantMeanProcess` or `ArProcess`

[`ConstantMeanProcess(mean=0.0)`][tail_lib.simulation.ConstantMeanProcess],
the same mean on every date. It is the default, with a zero mean.

[`ArProcess(ar, const=0.0)`][tail_lib.simulation.ArProcess], an AR(p)
mean, with at least one coefficient:

\[
\mu_t = \text{const} + \phi_1 r_{t-1} + \cdots + \phi_p r_{t-p}
\]

Before the first date, past returns are at the long-run mean
\(\text{const} / (1 - \sum \phi)\). Stationarity is not checked.

### Volatility: `ConstantVolProcess`, `GarchProcess` or `EwmaProcess`

[`ConstantVolProcess(vol)`][tail_lib.simulation.ConstantVolProcess], the
same positive volatility on every date. With Gaussian shocks and a constant
mean, the returns are i.i.d. Gaussian.

[`GarchProcess(omega, alpha, beta)`][tail_lib.simulation.GarchProcess], a
GARCH(1, 1) variance (Bollerslev, 1986):

\[
\sigma^2_{t+1} = \omega + \alpha \, \varepsilon_t^2 + \beta \, \sigma^2_t
\]

It starts from its long-run variance \(\omega / (1 - \alpha - \beta)\), so
\(\alpha + \beta\) must be below 1. To give the long-run variance instead of
\(\omega\):

```python
GarchProcess.from_long_run_variance(alpha=0.08, beta=0.90, long_run_variance=0.01**2)
```

[`EwmaProcess(decay, start_vol)`][tail_lib.simulation.EwmaProcess], the
RiskMetrics recursion from a starting volatility:

\[
\sigma^2_{t+1} = \lambda \, \sigma^2_t + (1 - \lambda) \, \varepsilon_t^2
\]

The EWMA has no long-run variance: on long paths it can drift to zero or
explode. Check the paths it gives.

### Shocks: a `Distribution`

Shocks are drawn from a distribution with zero mean and unit variance:

- [`Gaussian()`][tail_lib.distributions.Gaussian], the default;
- [`StudentT(df)`][tail_lib.distributions.StudentT], scaled to unit variance,
  with `df` above 2. The lower `df`, the fatter the tails.

### Your own process

Subclass [`MeanProcess`][tail_lib.simulation.MeanProcess] or
[`VolatilityProcess`][tail_lib.simulation.VolatilityProcess]. A process
holds its parameters, and its state is an array it passes in and out:
`start(batch_shape)` gives the state before the first date, `value(state)`
the mean or volatility of the next date, and `update(state, x)` moves one
date forward, with the return for a mean and the residual for a volatility.
`check_shape(batch_shape)` checks the parameters broadcast to the batch.

## `simulate` settings

| Argument | Meaning |
|---|---|
| `n_dates` | Number of dates in the output. |
| `first_date` | First date of the output; later dates follow the calendar. |
| `batch_shape` | Shape of each date's returns, e.g. `(n_paths,)`. Default `()`, one path. |
| `burn_in` | Dates simulated then dropped before the first date, so paths start away from the starting values. Default `0`. |
| `calendar` | The [`Calendar`][tail_lib.calendar.Calendar] of valid dates. Default Monday to Friday. |
| `rng` | A `numpy.random.Generator`, or a seed. The same seed gives the same paths. |

Every parameter is a number or an array that broadcasts to `batch_shape`,
e.g. a different `alpha` per path.

## Checking a model on simulated returns

With a zero mean, an `EwmaVolatility` given the true decay and starting
volatility gives back the simulated volatilities exactly:

```python
import numpy as np

from tail_lib.context import Context
from tail_lib.models.ewma import ConstantDecay, StartingVol
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.simulation import EwmaProcess

sim = Simulator(vol=EwmaProcess(decay=0.94, start_vol=0.01)).simulate(
    500, first_date=datetime.date(2020, 1, 1), rng=0
)
with Context() as ctx:
    vol = EwmaVolatility(ConstantDecay(0.94), start=StartingVol(0.01))
np.allclose(ctx.run(sim.returns)[vol].values, sim.vols.values)  # True
```

!!! note
    This only holds with a zero mean: the simulator's variance recursion runs
    on the residuals \(r_t - \mu_t\), the volatility models on the raw
    returns.
