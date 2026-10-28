import datetime
from typing import Literal

import numpy as np
import pytest
from scipy import stats

from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.mean import ConstantMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import FilteredReturns
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import Trailing
from tests.helpers import FIRST_DATE
from tests.helpers import returns_series
from tests.helpers import run_node
from tests.helpers import simulated_returns


def test_gaussian_compute_matches_normal_quantile() -> None:
    means = TimeSeries.from_first_date(FIRST_DATE, values=np.array([0.0, 0.001]))
    vols = TimeSeries.from_first_date(FIRST_DATE, values=np.array([0.01, 0.02]))

    var = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01)).compute(
        means, vol=vols, mean=means
    )

    np.testing.assert_array_equal(var.dates, means.dates)
    np.testing.assert_allclose(
        var.values,
        stats.norm.ppf(0.01, loc=[0.0, 0.001], scale=[0.01, 0.02]),
    )


def test_gaussian_compute_rejects_mismatched_dates() -> None:
    means = TimeSeries.from_first_date(FIRST_DATE, values=np.zeros(3))
    vols = TimeSeries.from_first_date(
        FIRST_DATE + datetime.timedelta(days=1), values=np.ones(3)
    )
    model = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))

    with pytest.raises(ValueError, match="dates"):
        model.compute(means, vol=vols, mean=means)


def test_gaussian_compute_rejects_mismatched_shapes() -> None:
    means = TimeSeries.from_first_date(FIRST_DATE, values=np.zeros((3, 2)))
    vols = TimeSeries.from_first_date(FIRST_DATE, values=np.ones(3))
    model = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))

    with pytest.raises(ValueError, match="shapes"):
        model.compute(means, vol=vols, mean=means)


@pytest.mark.parametrize("vol", [0.0, -0.01])
def test_gaussian_compute_rejects_non_positive_vol(vol: float) -> None:
    means = TimeSeries.from_first_date(FIRST_DATE, values=np.zeros(2))
    vols = TimeSeries.from_first_date(FIRST_DATE, values=np.array([0.01, vol]))
    model = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))

    with pytest.raises(ValueError, match="Volatility has to be positive"):
        model.compute(means, vol=vols, mean=means)


def test_gaussian_compute_keeps_nan_vol() -> None:
    means = TimeSeries.from_first_date(FIRST_DATE, values=np.zeros(2))
    vols = TimeSeries.from_first_date(FIRST_DATE, values=np.array([np.nan, 0.01]))
    model = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))

    var = model.compute(means, vol=vols, mean=means)

    assert np.isnan(var.values[0])
    np.testing.assert_allclose(var.values[1], 0.01 * stats.norm.ppf(0.01))


def test_gaussian_run_uses_mean_and_vol_models() -> None:
    returns = returns_series(n_assets=2)
    model = GaussianModel(
        cf=0.99,
        mean=ConstantMean(np.array([0.001, 0.002])),
        vol=ConstantVolatility(np.array([0.01, 0.02])),
    )

    var = run_node(model, returns)

    np.testing.assert_array_equal(var.dates, returns.dates)
    expected = stats.norm.ppf(0.01, loc=[0.001, 0.002], scale=[0.01, 0.02])
    np.testing.assert_allclose(var.values, np.tile(expected, (returns.n_dates, 1)))


def test_historical_simulations_is_quantile_of_each_trailing_window() -> None:
    returns = returns_series(n=30, n_assets=2)
    r = returns.values

    var = run_node(
        HistoricalSimulationsModel(cf=0.9, window=Trailing(length=10)), returns
    )

    np.testing.assert_array_equal(var.dates, returns.dates)
    assert np.isnan(var.values[:9]).all()
    for t in range(9, 30):
        expected = np.quantile(r[t - 9 : t + 1], 0.1, axis=0)
        np.testing.assert_allclose(var.values[t], expected)


def test_historical_simulations_holds_value_until_next_window() -> None:
    returns = returns_series(n=30)
    r = returns.values

    var = run_node(
        HistoricalSimulationsModel(cf=0.9, window=Trailing(length=10, step=5)), returns
    )

    for t in range(9, 30):
        last_end = 9 + 5 * ((t - 9) // 5)
        expected = np.quantile(r[last_end - 9 : last_end + 1], 0.1)
        np.testing.assert_allclose(var.values[t], expected)


@pytest.mark.parametrize(
    ("interpolation", "numpy_method"), [("below", "lower"), ("above", "higher")]
)
def test_historical_simulations_interpolation(
    interpolation: Literal["below", "above"], numpy_method: Literal["lower", "higher"]
) -> None:
    returns = returns_series(n=20)

    var = run_node(
        HistoricalSimulationsModel(
            cf=0.9, window=Trailing(length=20), interpolation=interpolation
        ),
        returns,
    )

    expected = np.quantile(returns.values, 0.1, method=numpy_method)
    np.testing.assert_allclose(var.values[-1], expected)


def test_historical_on_last_window_is_the_last_rolling_value() -> None:
    returns = returns_series(n=300, n_assets=2)

    last = run_node(
        HistoricalSimulationsModel(cf=0.99, window=Last(length=100)), returns
    )
    rolling = HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100))

    assert np.isnan(last.values[:-1]).all()
    np.testing.assert_array_equal(
        last.values[-1], run_node(rolling, returns).values[-1]
    )


def _standardized(returns: np.ndarray, vols: np.ndarray) -> np.ndarray:
    z = np.full(returns.shape, np.nan)
    z[1:] = returns[1:] / vols[:-1]
    return z


def test_filtered_historical_matches_formula_after_ewma_warm_up() -> None:
    returns = simulated_returns(n=300)
    vol = EwmaVolatility(ConstantDecay(0.94))
    vols = run_node(vol, returns).values
    z = _standardized(returns.values, vols)

    var = run_node(
        FilteredHistoricalSimulationsModel(
            cf=0.95, filter=VolatilityFilter(vol), window=Trailing(length=50)
        ),
        returns,
    )

    # The backcast takes the first 20 returns, so the first standardized
    # return is on date 20, and the first window holds dates 20 to 69 (R16).
    assert np.isnan(z[:20]).all()
    assert np.isnan(var.values[:69]).all()
    assert not np.isnan(var.values[69:]).any()
    for t in range(69, 300):
        expected = vols[t] * np.quantile(z[t - 49 : t + 1], 0.05, axis=0)
        np.testing.assert_allclose(var.values[t], expected)


def test_filtered_historical_steps_from_first_standardized_return() -> None:
    returns = simulated_returns(n=300)
    vol = EwmaVolatility(ConstantDecay(0.94))
    vols = run_node(vol, returns).values
    z = _standardized(returns.values, vols)

    var = run_node(
        FilteredHistoricalSimulationsModel(
            cf=0.95,
            filter=VolatilityFilter(vol),
            window=Trailing(length=50, step=7),
        ),
        returns,
    )

    # The steps count from the first window, on dates 20 to 69: windows end on
    # dates 69, 76, ..., and each value is held until the next.
    assert not np.isnan(var.values[69:]).any()
    for t in range(69, 300):
        end = 69 + 7 * ((t - 69) // 7)
        expected = vols[end] * np.quantile(z[end - 49 : end + 1], 0.05, axis=0)
        np.testing.assert_allclose(var.values[t], expected)


def test_filtered_historical_with_constant_mean_and_vol_is_historical() -> None:
    # A constant shift and scale move the quantile with them, so they cancel.
    returns = returns_series(n=60, n_assets=2)
    fhs = FilteredHistoricalSimulationsModel(
        cf=0.9,
        filter=VolatilityFilter(
            ConstantVolatility(np.array([0.01, 0.03])),
            mean=ConstantMean(np.array([0.001, -0.002])),
        ),
        window=Trailing(length=20),
    )
    hs = HistoricalSimulationsModel(cf=0.9, window=Trailing(length=20))

    var = run_node(fhs, returns)

    # The first return has no forecast before it, so FHS starts a date later.
    assert np.isnan(var.values[:20]).all()
    assert not np.isnan(var.values[20:]).any()
    np.testing.assert_allclose(var.values[20:], run_node(hs, returns).values[20:])


def test_filtered_historical_on_last_window_is_the_last_rolling_value() -> None:
    returns = simulated_returns(n=300)
    vol_filter = VolatilityFilter(EwmaVolatility(ConstantDecay(0.94)))

    last = run_node(
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=vol_filter, window=Last(length=100)
        ),
        returns,
    )
    rolling = run_node(
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=vol_filter, window=Trailing(length=100)
        ),
        returns,
    )

    assert np.isnan(last.values[:-1]).all()
    np.testing.assert_allclose(last.values[-1], rolling.values[-1])


def test_filtered_historical_rescales_with_the_mean_and_vol_of_each_date() -> None:
    returns = returns_series(n=40, n_assets=2)
    dates = returns.dates
    rng = np.random.default_rng(3)
    filtered = TimeSeries(dates=dates, values=rng.standard_t(5, (40, 2)))
    means = TimeSeries(dates=dates, values=rng.normal(0, 1e-3, (40, 2)))
    vols = TimeSeries(dates=dates, values=rng.uniform(0.005, 0.02, (40, 2)))
    model = FilteredHistoricalSimulationsModel(
        cf=0.9,
        filter=VolatilityFilter(ConstantVolatility(0.01)),
        window=Trailing(length=10),
    )

    var = model.compute(
        returns,
        filter=FilteredReturns(standardized=filtered, mean=means, vol=vols),
    )

    z, m, v = filtered.values, means.values, vols.values
    assert np.isnan(var.values[:9]).all()
    for t in range(9, 40):
        expected = m[t] + v[t] * np.quantile(z[t - 9 : t + 1], 0.1, axis=0)
        np.testing.assert_allclose(var.values[t], expected)
