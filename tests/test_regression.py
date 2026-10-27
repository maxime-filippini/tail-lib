"""Pinned outputs of every node type and of a run with backtests.

The context API migration (see "Context API" in `docs/open-points.md`)
rewrites how nodes are declared and run, but must not change a number. This
test computes one node of each type, and the backtests of a context, on
fixed returns, and compares them with the arrays saved in
`data/regression.npz`.

Each step of the migration rewrites how the nodes are built and computed
for the new API, and keeps the keys and the saved arrays. A step that changes a
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

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.backtests.kupiec import KupiecResult
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.context import Context
from tail_lib.distributions import StudentT
from tail_lib.graph import Node
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import DecayModel
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
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
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
        "var_input": TimeSeries(dates=returns.dates, values=np.full(shape, -0.025)),
    }


# Every node is built inside a context, by the functions below, so it binds
# to it and is computed even if no model reads it. Models in a context are
# named: the names are the pinned keys.

_BACKTESTS = {
    "kupiec": Kupiec(),
    "kupiec_250": Kupiec(window=Trailing(length=250)),
    "kupiec_100_step": Kupiec(window=Trailing(length=100, step=10)),
    "kupiec_last": Kupiec(window=Last(length=250)),
}
"""Kupiec tests on three windows, run on every model with a history."""


def _models() -> None:
    """One VaR model of each type and window, each named."""
    mean = ConstantMean(np.array([1e-4, -2e-4]))
    ewma = EwmaVolatility(ConstantDecay(0.94))
    vol_filter = VolatilityFilter(ewma, mean=mean)
    calibration = EwmaDecayCalibration(Trailing(length=150, step=50))
    HistoricalSimulationsModel(cf=0.99, window=Trailing(length=250), name="hs")
    HistoricalSimulationsModel(
        cf=0.99, window=Trailing(length=250, step=5), name="hs_step"
    )
    HistoricalSimulationsModel(
        cf=0.975, window=Trailing(length=250), interpolation="below", name="hs_below"
    )
    HistoricalSimulationsModel(
        cf=0.975, window=Trailing(length=250), interpolation="above", name="hs_above"
    )
    FilteredHistoricalSimulationsModel(
        cf=0.99, filter=vol_filter, window=Trailing(length=250), name="fhs"
    )
    FilteredHistoricalSimulationsModel(
        cf=0.99, filter=vol_filter, window=Trailing(length=250, step=5), name="fhs_step"
    )
    GaussianModel(cf=0.99, vol=ewma, mean=mean, name="gaussian_ewma")
    GaussianModel(cf=0.99, vol=EwmaVolatility(calibration), name="gaussian_fitted")
    GaussianModel(cf=0.99, vol=VolatilityInput("vol_ext"), name="gaussian_input_vol")
    # Named after its input, as every run input is.
    ValueAtRiskInput("var_input", cf=0.99)


def _last_models() -> None:
    """Models with a `Last` window: one VaR, so they can't be backtested."""
    ewma = EwmaVolatility(ConstantDecay(0.94))
    HistoricalSimulationsModel(cf=0.99, window=Last(length=250), name="hs_last")
    FilteredHistoricalSimulationsModel(
        cf=0.99, filter=VolatilityFilter(ewma), window=Last(length=250), name="fhs_last"
    )


def _decays() -> dict[str, DecayModel]:
    """One decay model of each type and window, by key."""
    return {
        "decay.constant": ConstantDecay(np.array([0.9, 0.97])),
        "decay.calibration": EwmaDecayCalibration(Trailing(length=150, step=50)),
        "decay.calibration_last": EwmaDecayCalibration(Last(length=150)),
    }


def _filters() -> dict[str, VolatilityFilter]:
    """One volatility filter with each kind of mean, by key."""
    ewma = EwmaVolatility(ConstantDecay(0.94))
    return {
        "filter.zero_mean": VolatilityFilter(ewma),
        "filter.constant_mean": VolatilityFilter(
            ewma, mean=ConstantMean(np.array([1e-4, -2e-4]))
        ),
    }


def _preprocessors(gaussian_ewma: ValueAtRiskModel) -> dict[str, Node[TimeSeries]]:
    """One node of each other type with a time series output, by key."""
    ewma = EwmaVolatility(ConstantDecay(0.94))
    return {
        "mean.zero": ZeroMean(),
        "mean.constant": ConstantMean(np.array([1e-4, -2e-4])),
        "vol.constant": ConstantVolatility(np.array([0.01, 0.02])),
        "vol.input": VolatilityInput("vol_ext"),
        "vol.ewma_backcast": ewma,
        "vol.ewma_starting_vol": EwmaVolatility(
            ConstantDecay(np.array([0.9, 0.97])), start=StartingVol(0.01)
        ),
        "vol.fitted_ewma": EwmaVolatility(
            EwmaDecayCalibration(Trailing(length=150, step=50))
        ),
        "vol.fitted_ewma_last": EwmaVolatility(EwmaDecayCalibration(Last(length=150))),
        "overshootings.gaussian_ewma": Overshootings(gaussian_ewma),
    }


def _outputs(returns: TimeSeries) -> dict[str, np.ndarray]:
    """Every pinned output, by key: node values, then backtest fields."""
    with Context(backtests=_BACKTESTS) as ctx:
        _models()
        preprocessors = _preprocessors(ctx.models["gaussian_ewma"])
        decays = _decays()
        filters = _filters()
    with Context() as today:
        _last_models()
    inputs = _inputs(returns)
    results = ctx.run(returns, inputs)
    last = today.run(returns, inputs)

    outputs = {
        f"node.{key}": results[node].values for key, node in preprocessors.items()
    }
    for model_results in (results, last):
        outputs |= {
            f"node.var.{name}": var.values for name, var in model_results.models.items()
        }
    # A decay model's factors are pinned; its refits are tested in test_ewma.
    outputs |= {
        f"node.{key}": results[decay].values.values for key, decay in decays.items()
    }
    # A filter's standardized returns are pinned; its mean and volatility are
    # those of its inputs, pinned on their own.
    outputs |= {
        f"node.{key}": results[node].standardized.values
        for key, node in filters.items()
    }

    for model, tests in results.backtests.items():
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
