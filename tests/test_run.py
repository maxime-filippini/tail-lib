import numpy as np
import pytest

from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import overshootings
from tail_lib.build import Builder
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.run import Run
from tail_lib.run import RunSpec
from tail_lib.specs import HistoricalSimulationsSpec
from tail_lib.specs import TrailingSpec
from tail_lib.windows import Trailing
from tests.helpers import returns_series

_SPEC = """
{
  "models": {
    "gauss_99": {
      "type": "gaussian", "cf": 0.99, "vol": {"type": "input", "name": "vol"}
    },
    "hs_99": {"type": "historical", "cf": 0.99, "window": {"length": 100}},
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
    assert kupiec.n_overshootings == expected.n_overshootings
    assert kupiec.statistic == expected.statistic
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
                "hs": {"type": "historical", "cf": 0.99, "window": {"length": 50}}
            },
            "backtests": {"a": {"type": "kupiec"}, "b": {"type": "kupiec"}},
        }
    )

    result = Run(spec).run(returns_series(n=200))

    assert result.backtests["hs"]["a"] is result.backtests["hs"]["b"]


def test_a_run_is_reusable_on_other_returns() -> None:
    spec = RunSpec.model_validate(
        {"models": {"hs": {"type": "historical", "cf": 0.99, "window": {"length": 50}}}}
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
