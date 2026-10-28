import dataclasses

import numpy as np
import pytest

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.backtests.overshootings import overshootings
from tail_lib.context import Context
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityInput
from tail_lib.run import RunSpec
from tail_lib.specs import HistoricalSimulationsSpec
from tail_lib.specs import TrailingSpec
from tail_lib.specs import find_last_window
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import Trailing
from tests.helpers import returns_series
from tests.helpers import run_node

_SPEC = """
{
  "models": {
    "gauss_99": {
      "type": "gaussian", "cf": 0.99, "vol": {"type": "input", "name": "vol"}
    },
    "hs_99": {
      "type": "historical", "cf": 0.99,
      "window": {"type": "trailing", "length": 100}
    },
    "var_ext": {"type": "input", "name": "var_ext", "cf": 0.95}
  },
  "backtests": {"kupiec": {"type": "kupiec"}}
}
"""


def test_run_computes_models_and_their_backtests() -> None:
    returns = returns_series(n=300)
    hs = HistoricalSimulationsModel(cf=0.95, window=Trailing(length=50))
    inputs = {
        "vol": run_node(ConstantVolatility(0.01), returns),
        "var_ext": run_node(hs, returns),
    }

    result = Context.from_spec(RunSpec.model_validate_json(_SPEC)).run(returns, inputs)

    assert (
        set(result.models) == set(result.backtests) == {"gauss_99", "hs_99", "var_ext"}
    )
    # The input model is backtested at its declared confidence level.
    expected = kupiec_pof(overshootings(inputs["var_ext"], returns), cf_var=0.95)
    kupiec = result.backtests["var_ext"]["kupiec"]
    np.testing.assert_array_equal(kupiec.n_overshootings, expected.n_overshootings)
    np.testing.assert_array_equal(kupiec.statistic, expected.statistic)
    np.testing.assert_array_equal(
        result.models["var_ext"].values, inputs["var_ext"].values
    )


def test_missing_inputs_fail_before_computing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> None:
        pytest.fail("A model was computed before the inputs were checked.")

    monkeypatch.setattr(HistoricalSimulationsModel, "compute", fail)
    run = Context.from_spec(RunSpec.model_validate_json(_SPEC))

    with pytest.raises(KeyError, match=r"Missing inputs \['var_ext', 'vol'\]"):
        run.run(returns_series(n=300), inputs={"other": returns_series(n=300)})


def test_input_declared_twice_differently_is_an_error() -> None:
    spec = RunSpec.model_validate(
        {
            "models": {
                "a": {"type": "input", "name": "var_ext", "cf": 0.99},
                "b": {"type": "input", "name": "var_ext", "cf": 0.95},
            }
        }
    )

    with pytest.raises(ValueError, match="'var_ext' is declared twice"):
        Context.from_spec(spec)


def test_identical_backtests_under_two_names_give_the_same_result() -> None:
    spec = RunSpec.model_validate(
        {
            "models": {
                "hs": {
                    "type": "historical",
                    "cf": 0.99,
                    "window": {"type": "trailing", "length": 50},
                }
            },
            "backtests": {"a": {"type": "kupiec"}, "b": {"type": "kupiec"}},
        }
    )

    result = Context.from_spec(spec).run(returns_series(n=200))

    assert result.backtests["hs"]["a"] is result.backtests["hs"]["b"]


def test_rolling_backtests_run_side_by_side_on_shared_overshootings() -> None:
    spec = RunSpec.model_validate_json("""
    {
      "models": {
        "hs": {
          "type": "historical", "cf": 0.99,
          "window": {"type": "trailing", "length": 50}
        }
      },
      "backtests": {
        "kupiec": {"type": "kupiec"},
        "kupiec_100": {
          "type": "kupiec", "window": {"type": "trailing", "length": 100}
        },
        "kupiec_150": {
          "type": "kupiec", "window": {"type": "trailing", "length": 150, "step": 5}
        },
        "kupiec_last": {"type": "kupiec", "window": {"type": "last", "length": 250}}
      }
    }
    """)
    returns = returns_series(n=400)
    run = Context.from_spec(spec)

    result = run.run(returns)

    flags = overshootings(result.models["hs"], returns)
    for name, window in [
        ("kupiec_100", Trailing(length=100)),
        ("kupiec_150", Trailing(length=150, step=5)),
    ]:
        expected = kupiec_pof(flags, cf_var=0.99, window=window)
        got = result.backtests["hs"][name]
        np.testing.assert_array_equal(got.end_dates, expected.end_dates)
        np.testing.assert_array_equal(got.statistic, expected.statistic)
    last = result.backtests["hs"]["kupiec_last"]
    expected_last = kupiec_pof(flags, cf_var=0.99, window=Last(length=250))
    np.testing.assert_array_equal(last.end_dates, returns.dates[-1:])
    np.testing.assert_array_equal(last.statistic, expected_last.statistic)
    assert result.backtests["hs"]["kupiec"].p_value.shape == (1,)
    assert result.backtests["hs"]["kupiec_100"].p_value.shape == (400 - 50 - 99,)


def test_rolling_backtests_of_a_model_share_its_overshootings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    compute = Overshootings.compute

    def counting(
        self: Overshootings, returns: TimeSeries, /, *, var: TimeSeries
    ) -> TimeSeries:
        calls.append(self)
        return compute(self, returns, var=var)

    monkeypatch.setattr(Overshootings, "compute", counting)
    spec = RunSpec.model_validate(
        {
            "models": {
                "hs": {
                    "type": "historical",
                    "cf": 0.99,
                    "window": {"type": "trailing", "length": 50},
                }
            },
            "backtests": {
                "kupiec_100": {
                    "type": "kupiec",
                    "window": {"type": "trailing", "length": 100},
                },
                "kupiec_last": {
                    "type": "kupiec",
                    "window": {"type": "last", "length": 200},
                },
            },
        }
    )

    Context.from_spec(spec).run(returns_series(n=300))

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("model_window", "backtest_window", "path"),
    [
        ({"length": 50}, {"type": "trailing", "length": 100}, r"models\.hs\."),
        ({"type": "trailing", "length": 50}, {"length": 100}, r"backtests\.k\."),
    ],
)
def test_a_window_needs_its_type(
    model_window: dict, backtest_window: dict, path: str
) -> None:
    spec = {
        "models": {"hs": {"type": "historical", "cf": 0.99, "window": model_window}},
        "backtests": {"k": {"type": "kupiec", "window": backtest_window}},
    }

    with pytest.raises(ValueError, match=path + r".*window\n.*Unable to extract tag"):
        RunSpec.model_validate(spec)


_LAST = {"type": "last", "length": 100}
_GAUSS_ON_LAST_CALIBRATION = {
    "type": "gaussian",
    "cf": 0.99,
    "vol": {"type": "ewma", "decay": {"type": "ewma_calibration", "window": _LAST}},
}


def test_a_model_with_a_last_window_can_not_be_backtested() -> None:
    spec = {
        "models": {"today": _GAUSS_ON_LAST_CALIBRATION},
        "backtests": {"kupiec": {"type": "kupiec"}},
    }

    with pytest.raises(
        ValueError,
        match=r"Model 'today' has a `last` window, at "
        r"`models\.today\.vol\.decay\.window`.*\['kupiec'\]",
    ):
        RunSpec.model_validate(spec)


def test_a_last_window_in_a_filter_is_found_through_it() -> None:
    spec = {
        "models": {
            "fhs": {
                "type": "filtered_historical",
                "cf": 0.99,
                "filter": {"vol": _GAUSS_ON_LAST_CALIBRATION["vol"]},
                "window": {"type": "trailing", "length": 50},
            }
        },
        "backtests": {"kupiec": {"type": "kupiec"}},
    }

    with pytest.raises(ValueError, match=r"`models\.fhs\.filter\.vol\.decay\.window`"):
        RunSpec.model_validate(spec)


def test_a_run_without_backtests_computes_last_window_models() -> None:
    spec = RunSpec.model_validate(
        {
            "models": {
                "today": _GAUSS_ON_LAST_CALIBRATION,
                "hs_today": {"type": "historical", "cf": 0.99, "window": _LAST},
            }
        }
    )

    result = Context.from_spec(spec).run(returns_series(n=300))

    for var in result.models.values():
        assert np.isnan(var.values[:-1]).all()
        assert not np.isnan(var.values[-1])


def test_find_last_window_gives_none_without_one() -> None:
    model = HistoricalSimulationsSpec(cf=0.99, window=TrailingSpec(length=100))

    assert find_last_window(model) is None


_TRAILING = {"type": "trailing", "length": 100, "step": 5}

# Between them, the two runs hold every spec class, and lists of values.
_BACKTESTING_RUN = {
    "models": {
        "fitted": {
            "type": "gaussian",
            "cf": 0.99,
            "vol": {
                "type": "ewma",
                "decay": {
                    "type": "ewma_calibration",
                    "window": _TRAILING,
                    "bounds": [0.8, 0.99],
                    "start": {"type": "backcast", "window": 10},
                    "n_iter": 30,
                },
                "start": {"type": "backcast", "window": 10},
            },
            "mean": {"type": "constant", "mean": [0.0, 0.001]},
        },
        "ewma": {
            "type": "gaussian",
            "cf": 0.975,
            "vol": {
                "type": "ewma",
                "decay": {"type": "constant", "value": [0.94, 0.97]},
                "start": {"type": "starting_vol", "vol": 0.01},
            },
        },
        "constant": {
            "type": "gaussian",
            "cf": 0.95,
            "vol": {"type": "constant", "vol": [0.01, 0.02]},
        },
        "on_input": {
            "type": "gaussian",
            "cf": 0.99,
            "vol": {"type": "input", "name": "v"},
        },
        "hs": {
            "type": "historical",
            "cf": 0.99,
            "window": _TRAILING,
            "interpolation": "below",
        },
        "var_ext": {"type": "input", "name": "var_ext", "cf": 0.99},
        "fhs": {
            "type": "filtered_historical",
            "cf": 0.99,
            "filter": {
                "type": "volatility_filter",
                "vol": {"type": "constant", "vol": 0.01},
                "mean": {"type": "zero"},
            },
            "window": _TRAILING,
            "interpolation": "above",
        },
    },
    "backtests": {
        "kupiec": {"type": "kupiec"},
        "kupiec_250": {"type": "kupiec", "window": {"type": "trailing", "length": 250}},
        "kupiec_last": {"type": "kupiec", "window": {"type": "last", "length": 250}},
    },
    "labels": {"fhs": "Filtered HS 99%", "kupiec_250": "Kupiec, 250 days"},
}

_PRODUCTION_RUN = {
    "models": {
        "today": _GAUSS_ON_LAST_CALIBRATION,
        "hs_today": {"type": "historical", "cf": 0.99, "window": _LAST},
    }
}


@pytest.mark.parametrize("spec", [_BACKTESTING_RUN, _PRODUCTION_RUN])
def test_a_run_spec_round_trips_through_json(spec: dict) -> None:
    parsed = RunSpec.model_validate(spec)

    text = parsed.model_dump_json()

    assert RunSpec.model_validate_json(text) == parsed
    # Writing it again gives the same text: nothing is lost or added.
    assert RunSpec.model_validate_json(text).model_dump_json() == text


def test_a_run_is_reusable_on_other_returns() -> None:
    spec = RunSpec.model_validate(
        {
            "models": {
                "hs": {
                    "type": "historical",
                    "cf": 0.99,
                    "window": {"type": "trailing", "length": 50},
                }
            }
        }
    )
    run = Context.from_spec(spec)

    first = run.run(returns_series(n=200))
    second = run.run(returns_series(n=200, n_assets=3))

    assert first.models["hs"].values.shape == (200,)
    assert second.models["hs"].values.shape == (200, 3)
    assert first.backtests == {"hs": {}}


def test_a_run_needs_a_model() -> None:
    with pytest.raises(ValueError, match="models"):
        RunSpec.model_validate({"models": {}})


def test_a_context_from_nodes_gives_the_numbers_of_its_spec() -> None:
    returns = returns_series(n=300)
    inputs = {"vol": run_node(ConstantVolatility(0.01), returns)}
    spec = RunSpec.model_validate_json(_SPEC)
    with Context(backtests={"kupiec": Kupiec()}) as ctx:
        GaussianModel(cf=0.99, vol=VolatilityInput("vol"), name="gauss_99")
        hs = HistoricalSimulationsModel(cf=0.99, window=Trailing(100), name="hs_99")
        ValueAtRiskInput("var_ext", cf=0.95)
    inputs["var_ext"] = run_node(hs, returns)

    result = ctx.run(returns, inputs)
    expected = Context.from_spec(spec).run(returns, inputs)

    for name, var in expected.models.items():
        np.testing.assert_array_equal(result.models[name].values, var.values)
        np.testing.assert_array_equal(
            result.backtests[name]["kupiec"].statistic,
            expected.backtests[name]["kupiec"].statistic,
        )


def test_backtests_take_their_settings_through_a_declaration() -> None:
    returns = returns_series(n=300)
    with Context(backtests={"kupiec_100": Kupiec(window=Trailing(length=100))}) as ctx:
        hs = HistoricalSimulationsModel(cf=0.99, window=Trailing(50), name="hs")

    result = ctx.run(returns)
    expected = kupiec_pof(
        overshootings(run_node(hs, returns), returns),
        cf_var=0.99,
        window=Trailing(length=100),
    )
    np.testing.assert_array_equal(
        result.backtests["hs"]["kupiec_100"].statistic, expected.statistic
    )


@pytest.mark.parametrize(
    ("model", "path"),
    [
        (
            HistoricalSimulationsModel(cf=0.99, window=Last(length=100)),
            "models.today.window",
        ),
        (
            GaussianModel(
                cf=0.99,
                vol=EwmaVolatility(EwmaDecayCalibration(Last(length=100))),
            ),
            "models.today.vol.decay.window",
        ),
    ],
)
def test_a_context_rejects_backtests_on_a_last_window_model(
    model: GaussianModel | HistoricalSimulationsModel, path: str
) -> None:
    with Context(backtests={"kupiec": Kupiec()}) as ctx:
        dataclasses.replace(model, name="today")

    with pytest.raises(
        ValueError,
        match=rf"Model 'today' has a `last` window, at `{path}`.*\['kupiec'\]",
    ):
        ctx.graph()


@pytest.mark.parametrize(
    ("models", "backtests", "path"),
    [
        ({"gauss 99": "hs"}, {}, "models.gauss 99"),
        ({"hs": "hs"}, {"kupiec-250": {"type": "kupiec"}}, "backtests.kupiec-250"),
    ],
)
def test_a_run_spec_rejects_names_that_are_not_identifiers(
    models: dict, backtests: dict, path: str
) -> None:
    hs = {
        "type": "historical",
        "cf": 0.99,
        "window": {"type": "trailing", "length": 100},
    }
    spec = {"models": dict.fromkeys(models, hs), "backtests": backtests}

    with pytest.raises(ValueError, match="Python identifier") as error:
        RunSpec.model_validate(spec)

    assert path in str(error.value)


def test_a_run_spec_rejects_an_input_name_that_is_not_an_identifier() -> None:
    spec = {"models": {"ext": {"type": "input", "name": "var.ext", "cf": 0.99}}}

    with pytest.raises(ValueError, match="Python identifier") as error:
        RunSpec.model_validate(spec)

    assert "models.ext.input.name" in str(error.value)


def test_a_run_from_a_spec_has_its_labels() -> None:
    run = Context.from_spec(RunSpec.model_validate(_BACKTESTING_RUN))

    assert run.labels["fhs"] == "Filtered HS 99%"
    assert run.labels["kupiec_250"] == "Kupiec, 250 days"
    assert run.labels["hs"] == "hs"


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ({"gauss": "Gaussian"}, r"Labels \['gauss'\] name no model or backtest"),
        ({"hs": ""}, r"Labels of \['hs'\] are empty"),
    ],
)
def test_a_context_rejects_bad_labels(labels: dict[str, str], match: str) -> None:
    with Context(labels=labels) as ctx:
        HistoricalSimulationsModel(cf=0.99, window=Trailing(100), name="hs")

    with pytest.raises(ValueError, match=match):
        ctx.graph()


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"backtests": {"hs": {"type": "kupiec"}}}, "both a model and a backtest"),
        ({"labels": {"gauss": "Gaussian"}}, "name no model or backtest"),
        ({"labels": {"hs": ""}}, "labels.hs"),
    ],
)
def test_a_run_spec_rejects_shared_names_and_bad_labels(
    extra: dict, match: str
) -> None:
    window = {"type": "trailing", "length": 100}
    spec = {
        "models": {"hs": {"type": "historical", "cf": 0.99, "window": window}},
        **extra,
    }

    with pytest.raises(ValueError, match=match):
        RunSpec.model_validate(spec)
