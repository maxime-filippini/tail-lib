"""Pinned outputs of every node type and of a run with backtests.

The context API migration (see "Context API" in `docs/open-points.md`)
rewrites how nodes are declared and run, but must not change a number. This
test computes one node of each type, and the backtests of a run, on fixed
returns, and compares them with the arrays saved in `data/regression.npz`.

Each step of the migration rewrites `_nodes`, `_run` and `_outputs` for the
new API, and keeps the keys and the saved arrays. A step that changes a
number on purpose regenerates the file and says why:

    uv run python -m tests.test_regression

The returns are saved in the file too, so regenerating it only recomputes
the outputs, and the simulator can change without moving the pins.
"""

import dataclasses
import datetime
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from tail_lib.backtests.kupiec import KupiecResult
from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.distributions import StudentT
from tail_lib.graph import Node
from tail_lib.graph import Pool
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import StartingVol
from tail_lib.models.mean import ConstantMean
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import FittedEwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
from tail_lib.run import Run
from tail_lib.simulation import GarchProcess
from tail_lib.simulation import Simulator
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing

PATH = Path(__file__).parent / "data" / "regression.npz"

# Relative tolerance: loose enough for NumPy versions summing in a different
# order (the tox matrix), tight enough to catch any change of method.
RTOL = 1e-10


def _simulated_returns() -> TimeSeries:
    """Two series of GARCH returns with Student-t shocks, 600 dates."""
    simulator = Simulator(
        vol=GarchProcess(omega=2e-6, alpha=0.08, beta=0.90), shocks=StudentT(df=5)
    )
    sim = simulator.simulate(
        600, first_date=datetime.date(2020, 1, 1), batch_shape=(2,), rng=7
    )
    return sim.returns


def _inputs(returns: TimeSeries) -> dict[str, TimeSeries]:
    """The run inputs, constant on every date."""
    shape = returns.values.shape
    return {
        "vol_ext": TimeSeries(dates=returns.dates, values=np.full(shape, 0.012)),
        "var_ext": TimeSeries(dates=returns.dates, values=np.full(shape, -0.025)),
    }


def _models() -> dict[str, ValueAtRiskModel]:
    """One VaR model of each type and window, by name."""
    mean = ConstantMean(np.array([1e-4, -2e-4]))
    ewma = EwmaVolatility(ConstantDecay(0.94))
    vol_filter = VolatilityFilter(ewma, mean_model=mean)
    calibration = EwmaDecayCalibration(Trailing(length=150, step=50))
    return {
        "hs": HistoricalSimulationsModel(cf=0.99, window=Trailing(length=250)),
        "hs_step": HistoricalSimulationsModel(
            cf=0.99, window=Trailing(length=250, step=5)
        ),
        "hs_below": HistoricalSimulationsModel(
            cf=0.975, window=Trailing(length=250), interpolation="below"
        ),
        "hs_above": HistoricalSimulationsModel(
            cf=0.975, window=Trailing(length=250), interpolation="above"
        ),
        "fhs": FilteredHistoricalSimulationsModel(
            cf=0.99, filter=vol_filter, window=Trailing(length=250)
        ),
        "fhs_step": FilteredHistoricalSimulationsModel(
            cf=0.99, filter=vol_filter, window=Trailing(length=250, step=5)
        ),
        "gaussian_ewma": GaussianModel(cf=0.99, vol_model=ewma, mean_model=mean),
        "gaussian_fitted": GaussianModel(
            cf=0.99, vol_model=FittedEwmaVolatility(calibration)
        ),
        "gaussian_input_vol": GaussianModel(
            cf=0.99, vol_model=VolatilityInput("vol_ext")
        ),
        "var_input": ValueAtRiskInput("var_ext", cf=0.99),
    }


def _last_models() -> dict[str, ValueAtRiskModel]:
    """Models with a `Last` window: one VaR, so they can't be backtested."""
    ewma = EwmaVolatility(ConstantDecay(0.94))
    return {
        "hs_last": HistoricalSimulationsModel(cf=0.99, window=Last(length=250)),
        "fhs_last": FilteredHistoricalSimulationsModel(
            cf=0.99, filter=VolatilityFilter(ewma), window=Last(length=250)
        ),
    }


def _nodes() -> dict[str, Node[TimeSeries]]:
    """One node of each type with a time series output, by key."""
    models = _models()
    calibration = EwmaDecayCalibration(Trailing(length=150, step=50))
    calibration_last = EwmaDecayCalibration(Last(length=150))
    ewma = EwmaVolatility(ConstantDecay(0.94))
    preprocessors: dict[str, Node[TimeSeries]] = {
        "mean.zero": ZeroMean(),
        "mean.constant": ConstantMean(np.array([1e-4, -2e-4])),
        "decay.constant": ConstantDecay(np.array([0.9, 0.97])),
        "decay.calibration": calibration,
        "decay.calibration_last": calibration_last,
        "vol.constant": ConstantVolatility(np.array([0.01, 0.02])),
        "vol.input": VolatilityInput("vol_ext"),
        "vol.ewma_backcast": ewma,
        "vol.ewma_starting_vol": EwmaVolatility(
            ConstantDecay(np.array([0.9, 0.97])), start=StartingVol(0.01)
        ),
        "vol.fitted_ewma": FittedEwmaVolatility(calibration),
        "vol.fitted_ewma_last": FittedEwmaVolatility(calibration_last),
        "filter.zero_mean": VolatilityFilter(ewma),
        "filter.constant_mean": VolatilityFilter(
            ewma, mean_model=ConstantMean(np.array([1e-4, -2e-4]))
        ),
        "overshootings.gaussian_ewma": Overshootings(models["gaussian_ewma"]),
    }
    var_models = {f"var.{name}": model for name, model in models.items()}
    last = {f"var.{name}": model for name, model in _last_models().items()}
    return preprocessors | var_models | last


def _run() -> Run:
    """The models with a history, each with Kupiec tests on three windows."""
    return Run(
        _models(),
        backtests={
            "kupiec": KupiecTest,
            "kupiec_250": lambda ovs: KupiecTest(ovs, window=Trailing(length=250)),
            "kupiec_100_step": lambda ovs: KupiecTest(
                ovs, window=Trailing(length=100, step=10)
            ),
            "kupiec_last": lambda ovs: KupiecTest(ovs, window=Last(length=250)),
        },
    )


def _outputs(returns: TimeSeries) -> dict[str, np.ndarray]:
    """Every pinned output, by key: node values, then backtest fields."""
    inputs = _inputs(returns)
    nodes = Pool(_nodes()).run(returns, inputs)
    outputs = {f"node.{key}": series.values for key, series in nodes.items()}

    result = _run().run(returns, inputs)
    for model, tests in result.backtests.items():
        for test, kupiec in tests.items():
            for field in dataclasses.fields(KupiecResult):
                key = f"backtest.{model}.{test}.{field.name}"
                outputs[key] = np.asarray(getattr(kupiec, field.name))
    return outputs


def _load() -> dict[str, np.ndarray]:
    """The saved arrays, by key, or none if the file is missing."""
    if not PATH.exists():
        return {}
    with np.load(PATH) as pinned:
        return {key: pinned[key] for key in pinned.files}


def _pinned_returns(pinned: Mapping[str, np.ndarray]) -> TimeSeries:
    return TimeSeries(dates=pinned["returns.dates"], values=pinned["returns.values"])


_PINNED = _load()
_OUTPUT_KEYS = sorted(key for key in _PINNED if not key.startswith("returns."))


@pytest.fixture(scope="module")
def outputs() -> dict[str, np.ndarray]:
    if not _PINNED:
        pytest.fail(
            f"No pinned outputs at {PATH}: run `python -m tests.test_regression`"
        )
    return _outputs(_pinned_returns(_PINNED))


def test_every_output_is_pinned(outputs: dict[str, np.ndarray]) -> None:
    assert sorted(outputs) == _OUTPUT_KEYS


@pytest.mark.parametrize("key", _OUTPUT_KEYS)
def test_output_matches_pinned(key: str, outputs: dict[str, np.ndarray]) -> None:
    expected = _PINNED[key]
    actual = outputs[key]
    assert actual.shape == expected.shape
    if np.issubdtype(expected.dtype, np.datetime64):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)


def _write() -> None:
    """Save the outputs, on the saved returns if any, else on new ones."""
    pinned = _load()
    returns = _pinned_returns(pinned) if pinned else _simulated_returns()
    arrays = {"returns.dates": returns.dates, "returns.values": returns.values}
    arrays |= _outputs(returns)
    PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(PATH, allow_pickle=False, **arrays)


if __name__ == "__main__":
    _write()
