import numpy as np
import pytest

from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import overshootings
from tail_lib.context import Context
from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import DecayFactors
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import StartingVol
from tail_lib.models.mean import ConstantMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
from tail_lib.run import RunSpec
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tests.helpers import returns_series
from tests.helpers import run_node


@pytest.mark.parametrize(
    ("spec", "node"),
    [
        pytest.param(
            {
                "type": "historical",
                "cf": 0.95,
                "window": {"type": "trailing", "length": 30, "step": 5},
                "interpolation": "above",
            },
            HistoricalSimulationsModel(
                cf=0.95, window=Trailing(30, step=5), interpolation="above"
            ),
            id="historical_trailing",
        ),
        pytest.param(
            {
                "type": "historical",
                "cf": 0.99,
                "window": {"type": "last", "length": 40},
                "interpolation": "below",
            },
            HistoricalSimulationsModel(cf=0.99, window=Last(40), interpolation="below"),
            id="historical_last",
        ),
        pytest.param(
            {
                "type": "gaussian",
                "cf": 0.99,
                "vol": {"type": "constant", "vol": [0.01, 0.02]},
                "mean": {"type": "constant", "mean": [0.001, -0.002]},
            },
            GaussianModel(
                cf=0.99,
                vol=ConstantVolatility(np.array([0.01, 0.02])),
                mean=ConstantMean(np.array([0.001, -0.002])),
            ),
            id="batch_constants",
        ),
        pytest.param(
            {
                "type": "gaussian",
                "cf": 0.975,
                "vol": {
                    "type": "ewma",
                    "decay": {"type": "constant", "value": [0.9, 0.97]},
                    "start": {"type": "starting_vol", "vol": [0.01, 0.02]},
                },
            },
            GaussianModel(
                cf=0.975,
                vol=EwmaVolatility(
                    ConstantDecay(np.array([0.9, 0.97])),
                    start=StartingVol(np.array([0.01, 0.02])),
                ),
            ),
            id="ewma_batch_decay_and_start",
        ),
        pytest.param(
            {
                "type": "gaussian",
                "cf": 0.99,
                "vol": {
                    "type": "ewma",
                    "decay": {
                        "type": "ewma_calibration",
                        "window": {"type": "last", "length": 60},
                        "bounds": [0.85, 0.98],
                        "n_iter": 7,
                        "start": {"type": "backcast", "window": 10},
                    },
                    "start": {"type": "backcast", "window": 10},
                },
            },
            GaussianModel(
                cf=0.99,
                vol=EwmaVolatility(
                    EwmaDecayCalibration(
                        Last(60), bounds=(0.85, 0.98), n_iter=7, start=Backcast(10)
                    ),
                    start=Backcast(10),
                ),
            ),
            id="calibrated_ewma_last",
        ),
        pytest.param(
            {
                "type": "gaussian",
                "cf": 0.95,
                "vol": {
                    "type": "ewma",
                    "decay": {
                        "type": "ewma_calibration",
                        "window": {"type": "trailing", "length": 60, "step": 20},
                        "start": {"type": "starting_vol", "vol": 0.02},
                    },
                    "start": {"type": "starting_vol", "vol": 0.02},
                },
            },
            GaussianModel(
                cf=0.95,
                vol=EwmaVolatility(
                    EwmaDecayCalibration(
                        Trailing(60, step=20), start=StartingVol(0.02)
                    ),
                    start=StartingVol(0.02),
                ),
            ),
            id="calibrated_ewma_trailing",
        ),
        pytest.param(
            {
                "type": "filtered_historical",
                "cf": 0.95,
                "filter": {
                    "vol": {
                        "type": "ewma",
                        "decay": {"type": "constant", "value": 0.94},
                    },
                    "mean": {"type": "constant", "mean": 0.001},
                },
                "window": {"type": "trailing", "length": 30, "step": 5},
                "interpolation": "below",
            },
            FilteredHistoricalSimulationsModel(
                cf=0.95,
                filter=VolatilityFilter(
                    EwmaVolatility(ConstantDecay(0.94)), mean=ConstantMean(0.001)
                ),
                window=Trailing(30, step=5),
                interpolation="below",
            ),
            id="filtered_historical_trailing",
        ),
        pytest.param(
            {
                "type": "filtered_historical",
                "cf": 0.99,
                "filter": {"vol": {"type": "constant", "vol": 0.01}},
                "window": {"type": "last", "length": 30},
            },
            FilteredHistoricalSimulationsModel(
                cf=0.99,
                filter=VolatilityFilter(ConstantVolatility(0.01)),
                window=Last(30),
            ),
            id="filtered_historical_last",
        ),
        pytest.param(
            {"type": "input", "name": "model", "cf": 0.95},
            ValueAtRiskInput("model", cf=0.95),
            id="var_input",
        ),
        pytest.param(
            {"type": "gaussian", "cf": 0.99, "vol": {"type": "input", "name": "vol"}},
            GaussianModel(cf=0.99, vol=VolatilityInput("vol")),
            id="volatility_input",
        ),
    ],
)
def test_spec_builds_and_runs_like_python(
    spec: dict[str, object], node: ValueAtRiskModel
) -> None:
    run_spec = RunSpec.model_validate({"models": {"model": spec}})
    ctx = Context.from_spec(run_spec)
    returns = returns_series(n=160, n_assets=2)
    inputs = {"model": returns, "vol": run_node(ConstantVolatility(0.02), returns)}

    assert ctx.models["model"] == node
    assert ctx.models["model"].name == "model"
    result = ctx.run(returns, inputs)["model"]
    expected = run_node(node, returns, inputs)
    np.testing.assert_array_equal(result.dates, expected.dates)
    np.testing.assert_array_equal(result.values, expected.values)
    assert np.isfinite(result.values[-1]).all()


@pytest.mark.parametrize("n_assets", [None, 2])
def test_spec_shares_calibration_and_runs_each_backtest_with_its_settings(
    n_assets: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    vol = {
        "type": "ewma",
        "decay": {
            "type": "ewma_calibration",
            "window": {"type": "trailing", "length": 40, "step": 20},
        },
    }
    model = {"type": "gaussian", "cf": 0.99, "vol": vol}
    spec = RunSpec.model_validate(
        {
            "models": {
                "gauss": model,
                "alias": model,
                "var_ext": {"type": "input", "name": "var_ext", "cf": 0.95},
                "input_99": {
                    "type": "gaussian",
                    "cf": 0.99,
                    "vol": {"type": "input", "name": "vol_svc"},
                },
                "input_95": {
                    "type": "gaussian",
                    "cf": 0.95,
                    "vol": {"type": "input", "name": "vol_svc"},
                },
                "fhs": {
                    "type": "filtered_historical",
                    "cf": 0.95,
                    "filter": {"vol": vol},
                    "window": {"type": "trailing", "length": 30},
                },
            },
            "backtests": {
                "full": {"type": "kupiec"},
                "same": {"type": "kupiec"},
                "rolling": {
                    "type": "kupiec",
                    "window": {"type": "trailing", "length": 20, "step": 5},
                },
                "last": {"type": "kupiec", "window": {"type": "last", "length": 30}},
            },
            "labels": {"gauss": "Gaussian 99%", "rolling": "Kupiec, 20 days"},
        }
    )
    calls = 0
    compute = EwmaDecayCalibration.compute

    def count(self: EwmaDecayCalibration, returns: TimeSeries, /) -> DecayFactors:
        nonlocal calls
        calls += 1
        return compute(self, returns)

    monkeypatch.setattr(EwmaDecayCalibration, "compute", count)
    ctx = Context.from_spec(spec)
    assert calls == 0
    assert ctx.labels["gauss"] == "Gaussian 99%"
    assert ctx.labels["rolling"] == "Kupiec, 20 days"
    assert ctx.labels["fhs"] == "fhs"

    for run_number, n in enumerate((160, 180), start=1):
        returns = returns_series(n=n, n_assets=n_assets)
        inputs = {
            "var_ext": run_node(
                HistoricalSimulationsModel(cf=0.95, window=Trailing(30)), returns
            ),
            "vol_svc": run_node(ConstantVolatility(0.02), returns),
        }
        results = ctx.run(returns, inputs)

        assert calls == run_number
        assert results["alias"] is results["gauss"]
        assert results["var_ext"] is inputs["var_ext"]
        assert results[VolatilityInput("vol_svc")] is inputs["vol_svc"]
        assert set(results.backtests) == set(spec.models)
        for name, node in ctx.models.items():
            assert set(results.backtests[name]) == set(spec.backtests)
            flags = overshootings(results[name], returns)
            for test, window in {
                "full": None,
                "rolling": Trailing(20, step=5),
                "last": Last(30),
            }.items():
                expected = kupiec_pof(flags, cf_var=node.cf, window=window)
                actual = results.backtests[name][test]
                np.testing.assert_array_equal(actual.end_dates, expected.end_dates)
                np.testing.assert_array_equal(actual.statistic, expected.statistic)
            assert results.backtests[name]["same"] is results.backtests[name]["full"]


def test_spec_rejects_different_declarations_of_one_input() -> None:
    spec = RunSpec.model_validate(
        {
            "models": {
                "gauss": {
                    "type": "gaussian",
                    "cf": 0.99,
                    "vol": {"type": "input", "name": "shared"},
                },
                "shared": {"type": "input", "name": "shared", "cf": 0.95},
            }
        }
    )

    with pytest.raises(ValueError, match="'shared' is declared twice, differently"):
        Context.from_spec(spec)


def test_a_var_input_model_key_has_to_match_its_name() -> None:
    spec = RunSpec.model_validate(
        {"models": {"var_ext": {"type": "input", "name": "external", "cf": 0.99}}}
    )

    with pytest.raises(ValueError, match=r"models\.var_ext.*model key has to match"):
        Context.from_spec(spec)
    assert Context.current() is None


@pytest.mark.parametrize("valid", [True, False])
def test_building_from_a_spec_restores_the_surrounding_context(*, valid: bool) -> None:
    spec = RunSpec.model_validate(
        {
            "models": {
                "gauss": {
                    "type": "gaussian",
                    "cf": 0.99,
                    "vol": {"type": "constant", "vol": 0.01},
                },
                "var_ext": {
                    "type": "input",
                    "cf": 0.95,
                    "name": "var_ext" if valid else "wrong",
                },
            }
        }
    )
    with Context() as outer:
        first = ConstantVolatility(0.02)
        if valid:
            Context.from_spec(spec)
        else:
            with pytest.raises(ValueError, match="model key has to match"):
                Context.from_spec(spec)
        assert Context.current() is outer
        last = ConstantVolatility(0.03)

    assert {entry.node for entry in outer.graph().nodes} == {first, last}
