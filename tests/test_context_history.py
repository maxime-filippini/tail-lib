import numpy as np
import pytest

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.context import Context
from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import FIRST_DATE
from tests.helpers import returns_series


@pytest.mark.parametrize("shape", [(60,), (60, 2), (60, 2, 3)])
def test_every_portfolio_starts_on_the_first_complete_date(
    shape: tuple[int, ...],
) -> None:
    values = np.random.default_rng(0).normal(scale=0.01, size=shape)
    values[:3] = np.nan
    if len(shape) > 1:
        values[:7, -1] = np.nan
    first = 3 if len(shape) == 1 else 7
    returns = TimeSeries.from_first_date(FIRST_DATE, values=values)
    original = values.copy()
    with Context() as ctx:
        HistoricalSimulationsModel(cf=0.99, window=Trailing(8), name="hs")
        vol = EwmaVolatility(ConstantDecay(0.94), start=Backcast(4))
        GaussianModel(cf=0.99, vol=vol, name="gauss")

    result = ctx.run(returns)
    expected = ctx.run(returns[first:])

    np.testing.assert_array_equal(result.returns.dates, returns.dates[first:])
    np.testing.assert_array_equal(result.returns.values, original[first:])
    np.testing.assert_array_equal(result["hs"].values, expected["hs"].values)
    np.testing.assert_array_equal(result[vol].values, expected[vol].values)
    np.testing.assert_array_equal(result["gauss"].values, expected["gauss"].values)
    np.testing.assert_array_equal(returns.values, original)


def test_calibration_filter_and_backtests_use_the_same_trimmed_history() -> None:
    returns = returns_series(n=90, n_assets=2)
    returns.values[:2, 0] = np.nan
    returns.values[:7, 1] = np.nan
    with Context() as ctx:
        decay = EwmaDecayCalibration(Trailing(12, step=7), start=Backcast(4), n_iter=4)
        vol = EwmaVolatility(decay, start=Backcast(4))
        filtered = VolatilityFilter(vol)
        GaussianModel(cf=0.99, vol=vol, name="gauss")
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=filtered, window=Trailing(8), name="fhs"
        )
        Kupiec(name="rolling", window=Trailing(10, step=3))

    result = ctx.run(returns)
    expected = ctx.run(returns[7:])

    # The first fit uses the first 12 complete returns; subsequent fits
    # count their step from there, independently of the original date zero.
    refits = np.arange(11, returns.n_dates - 7, 7)
    np.testing.assert_array_equal(result[decay].refits, refits)
    np.testing.assert_array_equal(
        result[decay].values.dates[refits], returns.dates[7 + refits]
    )
    np.testing.assert_array_equal(result[vol].values, expected[vol].values)
    np.testing.assert_array_equal(
        result[filtered].standardized.values, expected[filtered].standardized.values
    )
    for name in ("gauss", "fhs"):
        np.testing.assert_array_equal(result[name].values, expected[name].values)
        actual_test = result.backtests[name]["rolling"]
        expected_test = expected.backtests[name]["rolling"]
        np.testing.assert_array_equal(actual_test.end_dates, expected_test.end_dates)
        np.testing.assert_array_equal(actual_test.n_obs, expected_test.n_obs)
        np.testing.assert_array_equal(actual_test.p_value, expected_test.p_value)


def test_external_inputs_are_trimmed_with_returns_and_keep_their_warm_up() -> None:
    returns = returns_series(n=50, n_assets=2)
    returns.values[:5, 1] = np.nan
    values = np.full(returns.values.shape, 0.02)
    values[:10] = np.nan
    external = TimeSeries(dates=returns.dates, values=values)
    with Context() as ctx:
        vol = VolatilityInput("vol")
        GaussianModel(cf=0.99, vol=vol, name="gauss")
        ValueAtRiskInput("var", cf=0.99)

    result = ctx.run(
        returns, {"vol": external, "var": external, "unused": external[:3]}
    )

    for node in (vol, ctx.models["var"]):
        np.testing.assert_array_equal(result[node].dates, returns.dates[5:])
        np.testing.assert_array_equal(result[node].values, values[5:])
    assert np.isnan(result["gauss"].values[:5]).all()
    assert np.isfinite(result["gauss"].values[5:]).all()
    assert external.n_dates == returns.n_dates


def test_inputs_must_match_the_original_grid_before_trimming() -> None:
    returns = returns_series(n=50)
    returns.values[:5] = np.nan
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=VolatilityInput("vol"), name="gauss")

    with pytest.raises(ValueError, match="Input 'vol' does not match the returns"):
        ctx.run(returns, {"vol": returns[5:]})


def test_invalid_input_fails_before_any_node_computes() -> None:
    class _Counted(ConstantVolatility):
        n_calls = 0

        def compute(self, returns: TimeSeries, /) -> TimeSeries:
            _Counted.n_calls += 1
            return super().compute(returns)

    returns = returns_series(n=50)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_Counted(0.01), name="gauss")
        ValueAtRiskInput("var", cf=0.99)

    with pytest.raises(ValueError, match="Input 'var' does not match the returns"):
        ctx.run(returns, {"var": returns[1:]})
    assert _Counted.n_calls == 0


@pytest.mark.parametrize(
    "values",
    [np.empty(0), np.full(5, np.nan), np.array([[0.01, np.nan], [np.nan, 0.02]])],
)
def test_no_complete_date_fails_before_computing(values: np.ndarray) -> None:
    class _Never(ConstantVolatility):
        def compute(self, returns: TimeSeries, /) -> TimeSeries:  # noqa: ARG002
            pytest.fail("No node should compute without a common starting date")

    returns = TimeSeries.from_first_date(FIRST_DATE, values=values)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_Never(0.01), name="gauss")

    with pytest.raises(ValueError, match="no date with a value for every portfolio"):
        ctx.run(returns)


def test_trimming_is_specific_to_each_execution_of_a_context() -> None:
    complete = returns_series(n=50)
    late = TimeSeries(dates=complete.dates, values=complete.values.copy())
    late.values[:5] = np.nan
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")

    first = ctx.run(late)
    second = ctx.run(complete)

    assert first.returns.n_dates == complete.n_dates - 5
    assert second.returns is complete
    assert second["gauss"].n_dates == complete.n_dates
    assert first["gauss"].n_dates == first.returns.n_dates


def test_missing_observations_inside_history_are_left_in_place() -> None:
    returns = returns_series(n=50)
    returns.values[:5] = np.nan
    returns.values[20] = np.nan
    with Context() as ctx:
        HistoricalSimulationsModel(cf=0.99, window=Trailing(3), name="hs")

    result = ctx.run(returns)

    assert result.returns.n_dates == returns.n_dates - 5
    assert np.isnan(result.returns.values[15])
    assert np.isnan(result["hs"].values[15:18]).all()
