"""Backtest historical and filtered historical simulation with a rolling Kupiec test.

Models, all at 99%:

- historical simulation (HS) on 500 and 750 days;
- filtered historical simulation (FHS) on 500 and 750 days, filtered by an
  EWMA volatility with decay 0.94;
- a Gaussian model with the same EWMA volatility, for contrast.

Each model is backtested with Kupiec's test on rolling 250-day windows.

The library has no FHS model yet (see "Returns as a node" in
`docs/open-points.md`), so step 2 computes FHS from library nodes, and the run
reads it as a named input (R3, first mode; R13). Everything else is a spec.

Run with `uv run python examples/backtest_hs_fhs.py`.
"""

from __future__ import annotations

import datetime

import numpy as np

from tail_lib.distributions import StudentT
from tail_lib.graph import Cache
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.run import Run
from tail_lib.run import RunSpec
from tail_lib.simulation import GarchProcess
from tail_lib.simulation import Simulator
from tail_lib.specs import ConstantDecaySpec
from tail_lib.specs import EwmaVolatilitySpec
from tail_lib.specs import GaussianSpec
from tail_lib.specs import HistoricalSimulationsSpec
from tail_lib.specs import KupiecSpec
from tail_lib.specs import TrailingSpec
from tail_lib.specs import ValueAtRiskInputSpec
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing

CF = 0.99
DECAY = 0.94
HS_LENGTHS = (500, 750)
KUPIEC_LENGTH = 250
SIGNIFICANCE = 0.05


# Step 1: simulated returns.


def simulate_garch_returns(n: int, seed: int = 42) -> TimeSeries:
    """Simulate GARCH(1, 1) returns with Student-t shocks.

    The variance follows `var[t] = omega + alpha * r[t-1]**2 + beta * var[t-1]`,
    with `alpha = 0.08`, `beta = 0.90` and `omega` set for a long-run daily
    volatility of 1%, the variance of the first date. The shocks are
    Student-t with 5 degrees of freedom, scaled to unit variance, so the
    returns have volatility clusters and fat tails, as market returns do.
    """
    alpha, beta = 0.08, 0.90
    omega = 0.01**2 * (1 - alpha - beta)
    simulator = Simulator(
        vol=GarchProcess(omega=omega, alpha=alpha, beta=beta),
        shocks=StudentT(df=5.0),
    )
    sim = simulator.simulate(n, first_date=datetime.date(2015, 1, 1), rng=seed)
    return sim.returns


# Step 2: filtered historical simulation, from library nodes.


def fhs_var(
    filtered: TimeSeries, vols: TimeSeries, *, cf: float, length: int
) -> TimeSeries:
    """Compute the filtered historical simulation (FHS) value-at-risk.

    With `z[s] = r[s] / vol[s - 1]` the filtered returns, the value-at-risk
    on date `t` is the HS quantile of the last `length` filtered returns,
    rescaled by the volatility forecast for `t + 1`:

        VaR[t] = vol[t] * Q_{1 - cf}(z[t - length + 1], ..., z[t])

    It uses data up to and including `t` (D1). The mean is zero. The first value
    is on the `length`-th filtered return (R16).

    References: Hull and White, "Incorporating volatility updating into the
    historical simulation method for value-at-risk", Journal of Risk, 1998;
    Barone-Adesi, Giannopoulos and Vosper, "VaR without correlations for
    portfolios of derivative securities", Journal of Futures Markets, 1999.
    """
    hs = HistoricalSimulationsModel(cf=cf, window=Trailing(length=length))
    quantiles = hs.compute(filtered)
    return TimeSeries(dates=vols.dates, values=vols.values * quantiles.values)


def main() -> None:  # noqa: D103
    returns = simulate_garch_returns(n=2520)
    print(
        f"Step 1: {returns.n_dates} simulated returns, "
        f"{returns.dates[0]} to {returns.dates[-1]}, "
        f"daily vol {returns.values.std():.2%}."
    )

    # One cache, so the filter reuses the volatility computed for it.
    vol = EwmaVolatility(ConstantDecay(DECAY))
    cache = Cache(returns)
    vols = cache.get(vol)
    filtered = cache.get(VolatilityFilter(vol))
    fhs = {f"fhs_{n}": fhs_var(filtered, vols, cf=CF, length=n) for n in HS_LENGTHS}
    print(
        f"Step 2: EWMA vol from {_first_value_date(vols)}, filtered returns "
        f"from {_first_value_date(filtered)}, FHS VaR from "
        + ", ".join(f"{name} {_first_value_date(v)}" for name, v in fhs.items())
        + "."
    )

    # Step 3: the run spec. FHS models are named inputs, read from `fhs`.
    hs_specs = {
        f"hs_{n}": HistoricalSimulationsSpec(cf=CF, window=TrailingSpec(length=n))
        for n in HS_LENGTHS
    }
    fhs_specs = {name: ValueAtRiskInputSpec(name=name, cf=CF) for name in fhs}
    gauss_spec = GaussianSpec(
        cf=CF, vol=EwmaVolatilitySpec(decay=ConstantDecaySpec(value=DECAY))
    )
    spec = RunSpec(
        models=hs_specs | fhs_specs | {"gauss_ewma": gauss_spec},
        backtests={"kupiec_250": KupiecSpec(window=TrailingSpec(length=KUPIEC_LENGTH))},
    )
    run = Run(spec)
    print(f"Step 3: {len(spec.models)} models, run inputs {sorted(run.input_names)}.")

    # Step 4: compute the models and their backtests.
    result = run.run(returns, inputs=fhs)
    print("Step 4: run computed.")

    # Step 5: compare the models on the windows they all have (R4, R5).
    tests = {name: bt["kupiec_250"] for name, bt in result.backtests.items()}
    common = sorted(set.intersection(*(set(t.end_dates) for t in tests.values())))
    # The common windows test the returns from `first` on, against the VaR of
    # the day before each return (D1).
    first = int(np.searchsorted(returns.dates, common[0])) - KUPIEC_LENGTH + 1
    print(
        f"Step 5: {len(common)} common Kupiec windows, ending {common[0]} to "
        f"{common[-1]}. They test the returns from {returns.dates[first]}.\n"
    )

    expected = (1 - CF) * KUPIEC_LENGTH
    header = ("model", "mean VaR", "overshootings", "max", "rejected", "last p")
    units = ("", "", f"(exp. {expected:.1f})", "", f"(p < {SIGNIFICANCE})", "")
    for row in (header, units):
        print(f"{row[0]:<12}" + "".join(f"{c:>14}" for c in row[1:]))
    for name, test in tests.items():
        keep = np.isin(test.end_dates, common)
        n_over = test.n_overshootings[keep]
        p_value = test.p_value[keep]
        var = result.models[name].values[first - 1 : -1]
        cells = (
            f"{var.mean():.2%}",
            f"{n_over.mean():.2f}",
            f"{n_over.max()}",
            f"{(p_value < SIGNIFICANCE).mean():.0%}",
            f"{p_value[-1]:.3f}",
        )
        print(f"{name:<12}" + "".join(f"{c:>14}" for c in cells))


def _first_value_date(series: TimeSeries) -> datetime.date:
    first = int(np.argmax(~np.isnan(series.values)))
    return series.dates[first]


if __name__ == "__main__":
    main()
