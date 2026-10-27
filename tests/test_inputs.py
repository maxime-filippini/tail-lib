import numpy as np
import pytest

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import overshootings
from tail_lib.context import Context
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import VolatilityInput
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import returns_series
from tests.helpers import run_node


def test_var_input_is_backtested_like_the_model_it_came_from() -> None:
    returns = returns_series(n=300)
    hs = HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100))
    precomputed = run_node(hs, returns)

    with Context() as ctx:
        ValueAtRiskInput("var_99", cf=0.99)
        Kupiec(name="kupiec")

    from_input = ctx.run(returns, inputs={"var_99": precomputed}).backtests["var_99"][
        "kupiec"
    ]

    expected = kupiec_pof(overshootings(precomputed, returns), cf_var=0.99)
    np.testing.assert_array_equal(from_input.n_overshootings, expected.n_overshootings)
    np.testing.assert_array_equal(from_input.statistic, expected.statistic)


def test_volatility_input_feeds_a_model() -> None:
    returns = returns_series(n=50, n_assets=2)
    vols = run_node(ConstantVolatility(np.array([0.01, 0.02])), returns)

    model = GaussianModel(cf=0.99, vol=VolatilityInput("vol"))
    from_input = run_node(model, returns, inputs={"vol": vols})

    expected = run_node(
        GaussianModel(cf=0.99, vol=ConstantVolatility(np.array([0.01, 0.02]))), returns
    )
    np.testing.assert_array_equal(from_input.values, expected.values)


def test_context_run_passes_inputs_and_ignores_unused_ones() -> None:
    returns = returns_series(n=50)
    vols = run_node(ConstantVolatility(0.01), returns)

    with Context() as ctx:
        GaussianModel(cf=0.99, vol=VolatilityInput("vol"), name="g")

    outputs = ctx.run(returns, inputs={"vol": vols, "unused": vols})

    assert set(outputs.models) == {"g"}


def test_missing_input_is_an_error() -> None:
    returns = returns_series(n=50)

    with pytest.raises(KeyError, match=r"Missing inputs \['var_99'\]"):
        run_node(
            ValueAtRiskInput("var_99", cf=0.99), returns, inputs={"var_95": returns}
        )


def test_misaligned_input_is_an_error() -> None:
    returns = returns_series(n=50)
    shorter = TimeSeries(dates=returns.dates[1:], values=returns.values[1:])

    with pytest.raises(ValueError, match="Input 'vol' does not match the returns"):
        run_node(VolatilityInput("vol"), returns, inputs={"vol": shorter})


def test_var_input_validates_its_confidence_level() -> None:
    with pytest.raises(ValueError, match="Confidence level"):
        ValueAtRiskInput("var", cf=1.5)


def test_input_names_are_identifiers() -> None:
    with pytest.raises(ValueError, match=r"Input name .* got 'var ext'"):
        ValueAtRiskInput("var ext", cf=0.99)
    with pytest.raises(ValueError, match=r"Input name .* got 'vol-ext'"):
        VolatilityInput("vol-ext")
