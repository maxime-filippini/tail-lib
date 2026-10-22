import numpy as np
import pytest

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import overshootings
from tail_lib.graph import Pool
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import VolatilityInput
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import returns_series


def test_var_input_is_backtested_like_the_model_it_came_from() -> None:
    returns = returns_series(n=300)
    hs = HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100))
    precomputed = hs.run(returns)

    from_input = KupiecTest(ValueAtRiskInput("var_99", cf=0.99)).run(
        returns, inputs={"var_99": precomputed}
    )

    expected = kupiec_pof(overshootings(precomputed, returns), cf_var=0.99)
    assert from_input.n_overshootings == expected.n_overshootings
    assert from_input.statistic == expected.statistic


def test_volatility_input_feeds_a_model() -> None:
    returns = returns_series(n=50, n_assets=2)
    vols = ConstantVolatility(np.array([0.01, 0.02])).run(returns)

    model = GaussianModel(cf=0.99, vol_model=VolatilityInput("vol"))
    from_input = model.run(returns, inputs={"vol": vols})

    expected = GaussianModel(
        cf=0.99, vol_model=ConstantVolatility(np.array([0.01, 0.02]))
    ).run(returns)
    np.testing.assert_array_equal(from_input.values, expected.values)


def test_pool_run_passes_inputs_and_ignores_unused_ones() -> None:
    returns = returns_series(n=50)
    vols = ConstantVolatility(0.01).run(returns)

    outputs = Pool({"g": GaussianModel(cf=0.99, vol_model=VolatilityInput("vol"))}).run(
        returns, inputs={"vol": vols, "unused": vols}
    )

    assert set(outputs) == {"g"}


def test_missing_input_is_an_error() -> None:
    returns = returns_series(n=50)

    with pytest.raises(KeyError, match="Missing input 'var_99'"):
        ValueAtRiskInput("var_99", cf=0.99).run(returns, inputs={"var_95": returns})


def test_misaligned_input_is_an_error() -> None:
    returns = returns_series(n=50)
    shorter = TimeSeries(dates=returns.dates[1:], values=returns.values[1:])

    with pytest.raises(ValueError, match="Input 'vol' does not match the returns"):
        VolatilityInput("vol").run(returns, inputs={"vol": shorter})


def test_var_input_validates_its_confidence_level() -> None:
    with pytest.raises(ValueError, match="Confidence level"):
        ValueAtRiskInput("var", cf=1.5)
