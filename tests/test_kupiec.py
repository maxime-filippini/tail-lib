import numpy as np
import pytest

from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.time_series import TimeSeries
from tests.helpers import FIRST_DATE


def _flags(n: int, x: int) -> TimeSeries:
    values = np.zeros(n)
    values[:x] = 1.0
    return TimeSeries.from_first_date(FIRST_DATE, values=values)


def test_kupiec_matches_hand_computed_statistic() -> None:
    # n = 250, x = 5, p = 0.01, observed rate 0.02:
    # LR = -2 * [245 ln 0.99 + 5 ln 0.01 - 245 ln 0.98 - 5 ln 0.02]
    n, x = 250, 5
    result = kupiec_pof(_flags(n, x), cf=0.99)

    assert result.n_obs == n
    assert result.n_overshootings == x
    assert result.expected_rate == pytest.approx(0.01)
    assert result.observed_rate == pytest.approx(0.02)
    assert result.statistic == pytest.approx(1.956809788230622)
    assert result.p_value == pytest.approx(0.1618549171960387)


def test_kupiec_without_overshootings() -> None:
    # With x = 0 the observed term is 0: LR = -2 n ln(1 - p).
    result = kupiec_pof(_flags(250, 0), cf=0.99)

    assert result.statistic == pytest.approx(-2 * 250 * np.log(0.99))


def test_kupiec_with_every_date_an_overshooting() -> None:
    # With x = n the observed term is 0: LR = -2 n ln(p).
    result = kupiec_pof(_flags(250, 250), cf=0.99)

    assert result.statistic == pytest.approx(-2 * 250 * np.log(0.01))
    assert result.p_value == pytest.approx(0.0)


def test_kupiec_per_series_leaves_out_nan_dates() -> None:
    first = _flags(250, 5).values
    second = np.r_[np.full(50, np.nan), _flags(200, 0).values]
    flags = TimeSeries.from_first_date(
        FIRST_DATE, values=np.stack([first, second], axis=1)
    )

    result = kupiec_pof(flags, cf=0.99)
    alone = kupiec_pof(_flags(200, 0), cf=0.99)

    np.testing.assert_array_equal(result.n_obs, [250, 200])
    np.testing.assert_array_equal(result.n_overshootings, [5, 0])
    np.testing.assert_allclose(result.statistic, [1.956809788230622, alone.statistic])


def test_kupiec_without_observations_is_nan() -> None:
    flags = TimeSeries.from_first_date(FIRST_DATE, values=np.full(3, np.nan))

    result = kupiec_pof(flags, cf=0.99)

    assert result.n_obs == 0
    assert np.isnan(result.statistic)
    assert np.isnan(result.p_value)


@pytest.mark.parametrize("cf", [0.0, 1.0])
def test_kupiec_rejects_confidence_outside_unit_interval(cf: float) -> None:
    with pytest.raises(ValueError, match=r"\(0,1\)"):
        kupiec_pof(_flags(10, 1), cf=cf)
