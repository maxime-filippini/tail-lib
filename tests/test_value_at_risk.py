import datetime
from typing import Literal

import numpy as np
import pytest
from scipy import stats

from tail_lib.models.mean import ConstantMean
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tests.helpers import FIRST_DATE
from tests.helpers import returns_series


def test_gaussian_compute_matches_normal_quantile() -> None:
    means = TimeSeries.from_first_date(FIRST_DATE, values=np.array([0.0, 0.001]))
    vols = TimeSeries.from_first_date(FIRST_DATE, values=np.array([0.01, 0.02]))

    var = GaussianModel(cf=0.99, vol_model=ConstantVolatility(0.01)).compute(
        means=means, vols=vols
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
    model = GaussianModel(cf=0.99, vol_model=ConstantVolatility(0.01))

    with pytest.raises(ValueError, match="dates"):
        model.compute(means=means, vols=vols)


def test_gaussian_compute_rejects_mismatched_shapes() -> None:
    means = TimeSeries.from_first_date(FIRST_DATE, values=np.zeros((3, 2)))
    vols = TimeSeries.from_first_date(FIRST_DATE, values=np.ones(3))
    model = GaussianModel(cf=0.99, vol_model=ConstantVolatility(0.01))

    with pytest.raises(ValueError, match="shapes"):
        model.compute(means=means, vols=vols)


def test_gaussian_run_uses_mean_and_vol_models() -> None:
    returns = returns_series(n_assets=2)
    model = GaussianModel(
        cf=0.99,
        mean_model=ConstantMean(np.array([0.001, 0.002])),
        vol_model=ConstantVolatility(np.array([0.01, 0.02])),
    )

    var = model.run(returns)

    np.testing.assert_array_equal(var.dates, returns.dates)
    expected = stats.norm.ppf(0.01, loc=[0.001, 0.002], scale=[0.01, 0.02])
    np.testing.assert_allclose(var.values, np.tile(expected, (returns.n_dates, 1)))


def test_historical_simulations_is_quantile_of_each_trailing_window() -> None:
    returns = returns_series(n=30, n_assets=2)
    r = returns.values

    var = HistoricalSimulationsModel(cf=0.9, window=Trailing(length=10)).run(returns)

    np.testing.assert_array_equal(var.dates, returns.dates)
    assert np.isnan(var.values[:9]).all()
    for t in range(9, 30):
        expected = np.quantile(r[t - 9 : t + 1], 0.1, axis=0)
        np.testing.assert_allclose(var.values[t], expected)


def test_historical_simulations_holds_value_until_next_window() -> None:
    returns = returns_series(n=30)
    r = returns.values

    var = HistoricalSimulationsModel(cf=0.9, window=Trailing(length=10, step=5)).run(
        returns
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

    var = HistoricalSimulationsModel(
        cf=0.9, window=Trailing(length=20), interpolation=interpolation
    ).run(returns)

    expected = np.quantile(returns.values, 0.1, method=numpy_method)
    np.testing.assert_allclose(var.values[-1], expected)


def test_historical_on_last_window_is_the_last_rolling_value() -> None:
    returns = returns_series(n=300, n_assets=2)

    last = HistoricalSimulationsModel(cf=0.99, window=Last(length=100)).run(returns)
    rolling = HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100))

    assert np.isnan(last.values[:-1]).all()
    np.testing.assert_array_equal(last.values[-1], rolling.run(returns).values[-1])
