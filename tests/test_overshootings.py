import datetime

import numpy as np
import pytest

from tail_lib.backtests.overshootings import overshootings
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.time_series import TimeSeries
from tests.helpers import FIRST_DATE
from tests.helpers import returns_series


def _series(values: list[float] | np.ndarray) -> TimeSeries:
    return TimeSeries.from_first_date(FIRST_DATE, values=np.asarray(values, float))


def test_return_is_compared_with_previous_var() -> None:
    var = _series([-1.0, -2.0, -1.0, -1.0])
    returns = _series([-5.0, -1.5, -1.5, 0.0])

    hits = overshootings(var, returns)

    # r[1] = -1.5 vs VaR[0] = -1: overshooting. r[2] = -1.5 vs VaR[1] = -2: none.
    np.testing.assert_array_equal(hits.dates, returns.dates)
    np.testing.assert_array_equal(hits.values, [np.nan, 1.0, 0.0, 0.0])


def test_return_equal_to_var_is_not_an_overshooting() -> None:
    hits = overshootings(_series([-1.0, -1.0]), _series([0.0, -1.0]))

    assert hits.values[1] == 0.0


def test_dates_without_previous_var_or_return_are_nan() -> None:
    var = _series([np.nan, np.nan, -1.0, -1.0, -1.0])
    returns = _series([-2.0, -2.0, -2.0, np.nan, -2.0])

    hits = overshootings(var, returns)

    np.testing.assert_array_equal(hits.values, [np.nan, np.nan, np.nan, np.nan, 1.0])


def test_overshootings_per_series() -> None:
    returns = returns_series(n=100, n_assets=3)
    var = GaussianModel(cf=0.95, vol_model=ConstantVolatility(0.01)).run(returns)

    hits = overshootings(var, returns)

    assert hits.values.shape == (100, 3)
    np.testing.assert_array_equal(hits.values[1:], returns.values[1:] < var.values[:-1])


def test_overshootings_reject_var_on_other_dates() -> None:
    var = TimeSeries.from_first_date(
        FIRST_DATE + datetime.timedelta(days=1), values=np.full(3, -1.0)
    )

    with pytest.raises(ValueError, match="dates"):
        overshootings(var, _series([0.0, 0.0, 0.0]))
