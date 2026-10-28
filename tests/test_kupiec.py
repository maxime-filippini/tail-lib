import dataclasses

import numpy as np
import pytest

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.backtests.overshootings import overshootings
from tail_lib.context import Context
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import Trailing
from tests.helpers import FIRST_DATE
from tests.helpers import returns_series
from tests.helpers import run_node


def _flags(n: int, x: int) -> TimeSeries:
    values = np.zeros(n)
    values[:x] = 1.0
    return TimeSeries.from_first_date(FIRST_DATE, values=values)


def test_kupiec_matches_hand_computed_statistic() -> None:
    # n = 250, x = 5, p = 0.01, observed rate 0.02:
    # LR = -2 * [245 ln 0.99 + 5 ln 0.01 - 245 ln 0.98 - 5 ln 0.02]
    n, x = 250, 5
    result = kupiec_pof(_flags(n, x), cf_var=0.99)

    assert result.n_obs[0] == n
    assert result.n_overshootings[0] == x
    assert result.expected_rate == pytest.approx(0.01)
    assert result.observed_rate[0] == pytest.approx(0.02)
    assert result.statistic[0] == pytest.approx(1.956809788230622)
    assert result.p_value[0] == pytest.approx(0.1618549171960387)


def test_kupiec_on_the_whole_sample_is_one_window_ending_on_the_last_date() -> None:
    flags = TimeSeries.from_first_date(FIRST_DATE, values=np.zeros((250, 3)))

    result = kupiec_pof(flags, cf_var=0.99)

    np.testing.assert_array_equal(result.end_dates, flags.dates[-1:])
    for field in ("n_obs", "n_overshootings", "observed_rate", "statistic", "p_value"):
        assert getattr(result, field).shape == (1, 3)


def test_kupiec_without_overshootings() -> None:
    # With x = 0 the observed term is 0: LR = -2 n ln(1 - p).
    result = kupiec_pof(_flags(250, 0), cf_var=0.99)

    assert result.statistic[0] == pytest.approx(-2 * 250 * np.log(0.99))


def test_kupiec_with_every_date_an_overshooting() -> None:
    # With x = n the observed term is 0: LR = -2 n ln(p).
    result = kupiec_pof(_flags(250, 250), cf_var=0.99)

    assert result.statistic[0] == pytest.approx(-2 * 250 * np.log(0.01))
    assert result.p_value[0] == pytest.approx(0.0)


def test_kupiec_per_series_leaves_out_nan_dates() -> None:
    first = _flags(250, 5).values
    second = np.r_[np.full(50, np.nan), _flags(200, 0).values]
    flags = TimeSeries.from_first_date(
        FIRST_DATE, values=np.stack([first, second], axis=1)
    )

    result = kupiec_pof(flags, cf_var=0.99)
    alone = kupiec_pof(_flags(200, 0), cf_var=0.99)

    np.testing.assert_array_equal(result.n_obs, [[250, 200]])
    np.testing.assert_array_equal(result.n_overshootings, [[5, 0]])
    np.testing.assert_allclose(
        result.statistic, [[1.956809788230622, alone.statistic[0]]]
    )


def test_kupiec_without_observations_is_nan() -> None:
    flags = TimeSeries.from_first_date(FIRST_DATE, values=np.full(3, np.nan))

    result = kupiec_pof(flags, cf_var=0.99)

    assert result.n_obs[0] == 0
    assert np.isnan(result.statistic[0])
    assert np.isnan(result.p_value[0])


def test_kupiec_on_windows_matches_the_test_on_each_window() -> None:
    # Two leading NaN flags (a warm-up): windows start on date 2.
    values = np.r_[np.nan, np.nan, (np.arange(20) % 7 == 0).astype(float)]
    flags = TimeSeries.from_first_date(FIRST_DATE, values=values)

    result = kupiec_pof(flags, cf_var=0.9, window=Trailing(length=6, step=3))

    ends = np.arange(2 + 6 - 1, 22, 3)  # 7, 10, ..., 19
    np.testing.assert_array_equal(result.end_dates, flags.dates[ends])
    np.testing.assert_array_equal(result.n_obs, np.full(ends.size, 6))
    for k, end in enumerate(ends):
        alone = kupiec_pof(flags[end - 5 : end + 1], cf_var=0.9)
        assert result.n_overshootings[k] == alone.n_overshootings[0]
        assert result.statistic[k] == pytest.approx(alone.statistic[0])
        assert result.p_value[k] == pytest.approx(alone.p_value[0])


def test_kupiec_on_windows_keeps_the_batch_axes() -> None:
    flags = TimeSeries.from_first_date(FIRST_DATE, values=np.zeros((30, 2)))

    result = kupiec_pof(flags, cf_var=0.99, window=Trailing(length=10))

    assert result.p_value.shape == (21, 2)
    assert result.end_dates.shape == (21,)


def test_kupiec_rejects_a_window_longer_than_the_flags() -> None:
    values = np.r_[np.nan, np.zeros(5)]
    flags = TimeSeries.from_first_date(FIRST_DATE, values=values)

    with pytest.raises(ValueError, match="longer"):
        kupiec_pof(flags, cf_var=0.99, window=Trailing(length=6))


def test_kupiec_on_the_last_dates_matches_the_test_on_them() -> None:
    values = np.r_[np.nan, (np.arange(30) % 9 == 0).astype(float)]
    flags = TimeSeries.from_first_date(FIRST_DATE, values=values)

    result = kupiec_pof(flags, cf_var=0.9, window=Last(length=12))

    alone = kupiec_pof(flags[-12:], cf_var=0.9)
    np.testing.assert_array_equal(result.end_dates, flags.dates[-1:])
    np.testing.assert_array_equal(result.n_obs, [12])
    np.testing.assert_allclose(result.p_value, alone.p_value)


@pytest.mark.parametrize("cf_var", [0.0, 1.0])
def test_kupiec_rejects_confidence_outside_unit_interval(cf_var: float) -> None:
    with pytest.raises(ValueError, match=r"\(0,1\)"):
        kupiec_pof(_flags(10, 1), cf_var=cf_var)


def test_kupiec_declaration_matches_functions() -> None:
    returns = returns_series(n=300, n_assets=2)
    with Context() as ctx:
        model = HistoricalSimulationsModel(cf=0.95, window=Trailing(100), name="hs")
        Kupiec(name="kupiec")

    result = ctx.run(returns).backtests["hs"]["kupiec"]

    expected = kupiec_pof(overshootings(run_node(model, returns), returns), cf_var=0.95)
    np.testing.assert_array_equal(result.n_obs, expected.n_obs)
    np.testing.assert_array_equal(result.n_overshootings, expected.n_overshootings)
    np.testing.assert_allclose(result.p_value, expected.p_value)


def test_kupiec_declaration_on_windows_matches_functions() -> None:
    returns = returns_series(n=300)
    model = HistoricalSimulationsModel(cf=0.95, window=Trailing(length=100))
    window = Trailing(length=50, step=10)

    ctx = Context(backtests={"kupiec": Kupiec(window=window)})
    ctx.bind(dataclasses.replace(model, name="hs"))
    result = ctx.run(returns).backtests["hs"]["kupiec"]

    flags = overshootings(run_node(model, returns), returns)
    expected = kupiec_pof(flags, cf_var=0.95, window=window)
    # HS gives its first VaR on date 99, so the first flag is on date 100.
    assert result.end_dates[0] == returns.dates[100 + 50 - 1]
    np.testing.assert_array_equal(result.end_dates, expected.end_dates)
    np.testing.assert_allclose(result.p_value, expected.p_value)


def test_kupiec_test_uses_the_model_confidence_level() -> None:
    with Context(backtests={"kupiec": Kupiec()}) as ctx:
        GaussianModel(cf=0.975, vol=ConstantVolatility(0.01), name="gauss")

    result = ctx.run(returns_series(n=50)).backtests["gauss"]["kupiec"]

    assert result.expected_rate == pytest.approx(0.025)


def test_backtests_reuse_var_computed_in_the_same_context() -> None:
    class _CountingGaussian(GaussianModel):
        n_calls = 0

        def compute(
            self, returns: TimeSeries, /, *, vol: TimeSeries, mean: TimeSeries
        ) -> TimeSeries:
            _CountingGaussian.n_calls += 1
            return super().compute(returns, vol=vol, mean=mean)

    with Context(backtests={"kupiec": Kupiec()}) as ctx:
        _CountingGaussian(cf=0.99, vol=ConstantVolatility(0.01), name="gaussian")
        HistoricalSimulationsModel(cf=0.99, window=Trailing(length=50), name="hs")
    n = 200
    results = ctx.run(returns_series(n=n))

    assert _CountingGaussian.n_calls == 1
    assert set(results.models) == set(results.backtests) == {"gaussian", "hs"}
    # The first date has no previous VaR to compare with.
    assert results.backtests["gaussian"]["kupiec"].n_obs[0] == n - 1


def test_backtests_of_a_model_share_its_overshootings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    compute = Overshootings.compute

    def count(
        self: Overshootings, returns: TimeSeries, /, *, var: TimeSeries
    ) -> TimeSeries:
        nonlocal calls
        calls += 1
        return compute(self, returns, var=var)

    monkeypatch.setattr(Overshootings, "compute", count)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
        Kupiec(name="full")

        Kupiec(name="rolling", window=Trailing(20))

    results = ctx.run(returns_series(n=50))

    assert calls == 1
    assert set(results.backtests["gauss"]) == {"full", "rolling"}
