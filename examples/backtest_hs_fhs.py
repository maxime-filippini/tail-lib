"""Backtest historical and filtered historical simulation with a rolling Kupiec test.

Models, all at 99%:

- historical simulation (HS) on 500 and 750 days;
- filtered historical simulation (FHS) on 500 and 750 days, filtered by an
  EWMA volatility with decay 0.94;
- a Gaussian model with the same EWMA volatility, for contrast.

Each model is backtested with Kupiec's test on rolling 250-day windows.

The models are nodes, built in Python and given to a `Run` (R17). Nodes
shared by several models are computed once (D3, D9): the EWMA volatility,
and the filter of both FHS models.

Run with `uv run python examples/backtest_hs_fhs.py`.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.distributions import StudentT
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.run import Run
from tail_lib.simulation import GarchProcess
from tail_lib.simulation import Simulator
from tail_lib.windows import Trailing

if TYPE_CHECKING:
    from tail_lib.time_series import TimeSeries

CF = 0.99
DECAY = 0.94
HS_LENGTHS = (500, 750)
KUPIEC_LENGTH = 250
SIGNIFICANCE = 0.05


# Step 1: simulated returns.


def simulate_garch_returns(n: int, seed: int = 42) -> TimeSeries:
    """Simulate GARCH(1, 1) returns with Student-t shocks.

    The variance follows `var[t] = omega + alpha * r[t-1]**2 + beta * var[t-1]`,
    with `alpha = 0.08`, `beta = 0.90` and a long-run daily volatility of 1%,
    the volatility of the first date, from which `omega` is set. The shocks are
    Student-t with 5 degrees of freedom, scaled to unit variance, so the
    returns have volatility clusters and fat tails, as market returns do.
    """
    vol = GarchProcess.from_long_run_variance(
        alpha=0.08, beta=0.90, long_run_variance=0.01**2
    )
    simulator = Simulator(vol=vol, shocks=StudentT(df=5.0))
    sim = simulator.simulate(n, first_date=datetime.date(2015, 1, 1), rng=seed)
    return sim.returns


def main() -> None:  # noqa: D103
    returns = simulate_garch_returns(n=2520)
    print(
        f"Step 1: {returns.n_dates} simulated returns, "
        f"{returns.dates[0]} to {returns.dates[-1]}, "
        f"daily vol {returns.values.std():.2%}."
    )

    # Step 2: the models, as nodes. The EWMA volatility (with its 20-day
    # backcast) is read by the filter and the Gaussian model; the filter is
    # read by both FHS models. Each is one node, computed once.
    vol = EwmaVolatility(ConstantDecay(DECAY))
    vol_filter = VolatilityFilter(vol)
    hs = {
        f"hs_{n}": HistoricalSimulationsModel(cf=CF, window=Trailing(length=n))
        for n in HS_LENGTHS
    }
    fhs = {
        f"fhs_{n}": FilteredHistoricalSimulationsModel(
            cf=CF, filter=vol_filter, window=Trailing(length=n)
        )
        for n in HS_LENGTHS
    }
    gauss = {"gauss_ewma": GaussianModel(cf=CF, vol=vol)}
    models = hs | fhs | gauss
    print(f"Step 2: {len(models)} models: {', '.join(models)}.")

    # Step 3: the run. Each backtest is a function from a model's
    # overshootings to the test, so every model gets the same test (R4).
    run = Run(
        models,
        backtests={
            "kupiec_250": lambda ovs: KupiecTest(
                ovs, window=Trailing(length=KUPIEC_LENGTH)
            )
        },
    )
    print(f"Step 3: run built, reading {len(run.input_names)} run inputs.")

    # Step 4: compute the models and their backtests.
    result = run.run(returns)
    print(
        "Step 4: run computed. First VaR: "
        + ", ".join(
            f"{name} {_first_value_date(var)}" for name, var in result.models.items()
        )
        + "."
    )

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
