import numpy as np
import pytest

from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.backtests.overshootings import overshootings
from tail_lib.build import Builder
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.run import Run
from tail_lib.run import RunSpec
from tail_lib.specs import HistoricalSimulationsSpec
from tail_lib.specs import TrailingSpec
from tail_lib.specs import find_last_window
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tests.helpers import returns_series

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
    "ext_95": {"type": "input", "name": "var_ext", "cf": 0.95}
  },
  "backtests": {"kupiec": {"type": "kupiec"}}
}
"""


def test_run_computes_models_and_their_backtests() -> None:
    returns = returns_series(n=300)
    hs = HistoricalSimulationsModel(cf=0.95, window=Trailing(length=50))
    inputs = {
        "vol": ConstantVolatility(0.01).run(returns),
        "var_ext": hs.run(returns),
    }

    result = Run(RunSpec.model_validate_json(_SPEC)).run(returns, inputs)

    assert (
        set(result.models) == set(result.backtests) == {"gauss_99", "hs_99", "ext_95"}
    )
    # The input model is backtested at its declared confidence level.
    expected = kupiec_pof(overshootings(inputs["var_ext"], returns), cf_var=0.95)
    kupiec = result.backtests["ext_95"]["kupiec"]
    np.testing.assert_array_equal(kupiec.n_overshootings, expected.n_overshootings)
    np.testing.assert_array_equal(kupiec.statistic, expected.statistic)
    np.testing.assert_array_equal(
        result.models["ext_95"].values, inputs["var_ext"].values
    )


def test_run_lists_the_inputs_its_specs_read() -> None:
    run = Run(RunSpec.model_validate_json(_SPEC))

    assert run.input_names == {"vol", "var_ext"}


def test_missing_inputs_fail_before_computing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> None:
        pytest.fail("A model was computed before the inputs were checked.")

    monkeypatch.setattr(HistoricalSimulationsModel, "compute", fail)
    run = Run(RunSpec.model_validate_json(_SPEC))

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
        Run(spec)


def test_backtests_of_a_model_share_its_overshootings() -> None:
    builder = Builder()
    model = HistoricalSimulationsSpec(cf=0.99, window=TrailingSpec(length=100))

    first = builder.overshootings(model)

    assert builder.overshootings(model.model_copy()) is first


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

    result = Run(spec).run(returns_series(n=200))

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
    run = Run(spec)

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
        self: Overshootings, returns: TimeSeries, *, var: TimeSeries
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

    Run(spec).run(returns_series(n=300))

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
    "vol": {"type": "fitted_ewma", "calibration": {"window": _LAST}},
}


def test_a_model_with_a_last_window_can_not_be_backtested() -> None:
    spec = {
        "models": {"today": _GAUSS_ON_LAST_CALIBRATION},
        "backtests": {"kupiec": {"type": "kupiec"}},
    }

    with pytest.raises(
        ValueError,
        match=r"Model 'today' has a `last` window, at "
        r"`models\.today\.vol\.calibration\.window`.*\['kupiec'\]",
    ):
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

    result = Run(spec).run(returns_series(n=300))

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
                "type": "fitted_ewma",
                "calibration": {
                    "type": "ewma_calibration",
                    "window": _TRAILING,
                    "bounds": [0.8, 0.99],
                    "start": {"type": "backcast", "window": 10},
                    "n_iter": 30,
                },
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
        "ext": {"type": "input", "name": "var_ext", "cf": 0.99},
    },
    "backtests": {
        "kupiec": {"type": "kupiec"},
        "kupiec_250": {"type": "kupiec", "window": {"type": "trailing", "length": 250}},
        "kupiec_last": {"type": "kupiec", "window": {"type": "last", "length": 250}},
    },
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
    run = Run(spec)

    first = run.run(returns_series(n=200))
    second = run.run(returns_series(n=200, n_assets=3))

    assert first.models["hs"].values.shape == (200,)
    assert second.models["hs"].values.shape == (200, 3)
    assert first.backtests == {"hs": {}}


def test_a_run_needs_a_model() -> None:
    with pytest.raises(ValueError, match="models"):
        RunSpec.model_validate({"models": {}})
