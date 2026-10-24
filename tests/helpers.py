"""Data and models shared by the tests."""

import datetime

import numpy as np

from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import StartingVol
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.simulation import EwmaProcess
from tail_lib.simulation import Simulator
from tail_lib.time_series import TimeSeries

FIRST_DATE = datetime.date(2020, 1, 1)


def returns_series(n: int = 250, n_assets: int | None = None) -> TimeSeries:
    rng = np.random.default_rng(0)
    shape = (n,) if n_assets is None else (n, n_assets)
    return TimeSeries.from_first_date(
        FIRST_DATE, values=rng.normal(scale=0.01, size=shape)
    )


def ewma(decay: np.ndarray | float, start: float = 0.01) -> EwmaVolatility:
    return EwmaVolatility(ConstantDecay(decay), start=StartingVol(start))


def simulate_ewma(decay_factor: np.ndarray, n: int) -> np.ndarray:
    simulator = Simulator(vol=EwmaProcess(decay=decay_factor, start_vol=0.01))
    sim = simulator.simulate(
        n, first_date=FIRST_DATE, batch_shape=decay_factor.shape, rng=1
    )
    return sim.returns.values


def simulated_returns(n: int = 160) -> TimeSeries:
    return TimeSeries.from_first_date(
        FIRST_DATE, values=simulate_ewma(np.array([0.9, 0.97]), n=n)
    )
