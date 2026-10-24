import datetime

import numpy as np
import pytest

from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import StartingVol
from tail_lib.models.mean import ConstantMean
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import FittedEwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing
from tests.helpers import FIRST_DATE
from tests.helpers import ewma
from tests.helpers import returns_series
from tests.helpers import simulated_returns


def test_ewma_volatility_recursion() -> None:
    returns = returns_series(n=5)
    vols = ewma(0.94).run(returns)

    # The volatility on date t is the forecast for t + 1: it uses r[t].
    expected = []
    previous = 0.01
    for r in returns.values:
        previous = np.sqrt(0.94 * previous**2 + 0.06 * r**2)
        expected.append(previous)

    np.testing.assert_array_equal(vols.dates, returns.dates)
    np.testing.assert_allclose(vols.values, expected)


def test_ewma_volatility_backcast_by_default() -> None:
    returns = returns_series(n=30)
    r = returns.values
    n = 20

    vols = EwmaVolatility(ConstantDecay(0.94)).run(returns)

    # The backcast uses dates 0 to n - 1, so it is the volatility as of n - 1.
    expected = [np.sqrt(np.mean(r[:n] ** 2))]
    for t in range(n, 30):
        expected.append(np.sqrt(0.94 * expected[-1] ** 2 + 0.06 * r[t] ** 2))

    assert np.isnan(vols.values[: n - 1]).all()
    np.testing.assert_allclose(vols.values[n - 1 :], expected)


def test_ewma_volatility_backcast_window_must_fit_returns() -> None:
    with pytest.raises(ValueError, match="Backcast window"):
        EwmaVolatility(ConstantDecay(0.94), start=Backcast(window=30)).run(
            returns_series(n=30)
        )


def test_ewma_volatility_with_batch_decay_factors() -> None:
    returns = returns_series(n=50, n_assets=2)
    decay = np.array([0.9, 0.97])

    vols = ewma(decay).run(returns)

    for i, d in enumerate(decay):
        single = TimeSeries(dates=returns.dates, values=returns.values[:, i])
        np.testing.assert_allclose(vols.values[:, i], ewma(d).run(single).values)


def test_ewma_volatility_rejects_decay_not_matching_batch() -> None:
    returns = returns_series(n=50, n_assets=2)
    model = ewma(np.array([0.9, 0.94, 0.97]))

    with pytest.raises(ValueError, match="broadcast"):
        model.run(returns)


def test_ewma_volatility_compute_matches_constant_decay_run() -> None:
    returns = returns_series(n=50, n_assets=2)
    decay = TimeSeries(dates=returns.dates, values=np.full((50, 2), 0.94))

    np.testing.assert_allclose(
        ewma(0.94).compute(returns, decay=decay).values,
        ewma(0.94).run(returns).values,
    )


def test_ewma_volatility_decay_applies_from_its_own_date() -> None:
    returns = returns_series(n=50)
    switch = 30
    decay = TimeSeries(
        dates=returns.dates,
        values=np.where(np.arange(50) < switch, 0.9, 0.97),
    )

    vols = ewma(0.9).compute(returns, decay=decay)
    constant = ewma(0.9).run(returns)

    np.testing.assert_allclose(vols.values[:switch], constant.values[:switch])
    assert not np.isclose(vols.values[switch], constant.values[switch])


def test_ewma_volatility_rejects_decay_series_on_other_dates() -> None:
    returns = returns_series(n=50)
    decay = TimeSeries.from_first_date(
        FIRST_DATE + datetime.timedelta(days=1), values=np.full(50, 0.94)
    )

    with pytest.raises(ValueError, match="dates"):
        ewma(0.94).compute(returns, decay=decay)


def test_fitted_ewma_volatility_starts_after_first_calibration() -> None:
    returns = simulated_returns()
    vol_model = FittedEwmaVolatility(
        EwmaDecayCalibration(Trailing(length=100, step=20))
    )

    vols = vol_model.run(returns)
    var = GaussianModel(cf=0.99, vol_model=vol_model).run(returns)

    # The first calibration is known on date 99, the end of its window.
    np.testing.assert_array_equal(vols.dates, returns.dates)
    assert np.isnan(vols.values[:99]).all()
    assert np.isfinite(vols.values[99:]).all()
    assert np.isnan(var.values[:99]).all()
    assert np.isfinite(var.values[99:]).all()


def test_fitted_ewma_volatility_has_no_look_ahead() -> None:
    returns = simulated_returns()
    vol_model = FittedEwmaVolatility(
        EwmaDecayCalibration(Trailing(length=100, step=20))
    )
    t = 130

    full = vol_model.run(returns)
    truncated = vol_model.run(returns[: t + 1])

    np.testing.assert_allclose(full.values[: t + 1], truncated.values)


def test_fitted_ewma_volatility_uses_calibrated_decay() -> None:
    returns = simulated_returns()
    calibration = EwmaDecayCalibration(Trailing(length=100, step=20))
    vol_model = FittedEwmaVolatility(calibration)

    decay = calibration.run(returns)

    np.testing.assert_allclose(
        vol_model.run(returns).values,
        vol_model.compute(
            returns, decay=decay, window=calibration.window, start=calibration.start
        ).values,
    )


def test_fitted_ewma_volatility_refilters_each_training_window() -> None:
    returns = simulated_returns()
    length, step = 100, 20
    calibration = EwmaDecayCalibration(Trailing(length=length, step=step))
    decay = calibration.run(returns)

    vols = FittedEwmaVolatility(calibration).run(returns)

    # From the end of training window k until the next fit, the volatility is
    # the EWMA with fit k, run from the start of the window, as if the returns
    # started there.
    for k, window_end in enumerate(range(length - 1, returns.n_dates, step)):
        window_start = window_end - length + 1
        fit = decay.values[window_end]
        refiltered = EwmaVolatility(ConstantDecay(fit)).run(returns[window_start:])
        dates = slice(window_end, min(window_end + step, returns.n_dates))
        np.testing.assert_allclose(
            vols.values[dates],
            refiltered.values[dates.start - window_start : dates.stop - window_start],
            err_msg=f"fit {k}",
        )


def test_fitted_ewma_volatility_uses_the_calibration_start() -> None:
    returns = simulated_returns()
    calibration = EwmaDecayCalibration(
        Trailing(length=100, step=20), start=StartingVol(0.01)
    )
    fit = calibration.run(returns).values[99]

    vols = FittedEwmaVolatility(calibration).run(returns)

    first_window = EwmaVolatility(ConstantDecay(fit), start=StartingVol(0.01))
    np.testing.assert_allclose(
        vols.values[99:119], first_window.run(returns[:119]).values[99:119]
    )


def test_volatility_filter_divides_returns_by_vols() -> None:
    returns = returns_series(n=50, n_assets=2)
    vol_model = ewma(0.94)

    filtered = VolatilityFilter(vol_model).run(returns)

    # The return of date t is divided by the forecast for t, made on t - 1.
    vols = vol_model.run(returns)
    assert np.isnan(filtered.values[0]).all()
    np.testing.assert_allclose(
        filtered.values[1:], returns.values[1:] / vols.values[:-1]
    )


def test_volatility_filter_takes_off_the_mean_forecast_for_each_date() -> None:
    returns = returns_series(n=50, n_assets=2)
    rng = np.random.default_rng(3)
    means = TimeSeries(dates=returns.dates, values=rng.normal(0, 1e-3, (50, 2)))
    vols = TimeSeries(dates=returns.dates, values=rng.uniform(0.005, 0.02, (50, 2)))

    filtered = VolatilityFilter(ConstantVolatility(0.01)).compute(
        returns, means=means, vols=vols
    )

    # The mean and volatility forecast for date t are those of date t - 1.
    assert np.isnan(filtered.values[0]).all()
    np.testing.assert_allclose(
        filtered.values[1:],
        (returns.values[1:] - means.values[:-1]) / vols.values[:-1],
    )


def test_volatility_filter_run_uses_its_mean_model() -> None:
    returns = returns_series(n=50, n_assets=2)
    mean = np.array([0.001, -0.002])

    filtered = VolatilityFilter(
        ConstantVolatility(0.01), mean_model=ConstantMean(mean)
    ).run(returns)

    np.testing.assert_allclose(filtered.values[1:], (returns.values[1:] - mean) / 0.01)


def test_volatility_filter_rejects_mismatched_vols() -> None:
    returns = returns_series(n=50)
    means = TimeSeries(dates=returns.dates, values=np.zeros(50))
    vols = TimeSeries.from_first_date(
        FIRST_DATE + datetime.timedelta(days=1), values=np.full(50, 0.01)
    )

    with pytest.raises(ValueError, match="dates"):
        VolatilityFilter(ConstantVolatility(0.01)).compute(
            returns, means=means, vols=vols
        )


def test_fitted_ewma_on_last_window_is_the_last_rolling_value() -> None:
    returns = simulated_returns(n=160)

    last = FittedEwmaVolatility(EwmaDecayCalibration(Last(length=100))).run(returns)
    rolling = FittedEwmaVolatility(EwmaDecayCalibration(Trailing(length=100)))

    assert np.isnan(last.values[:-1]).all()
    np.testing.assert_allclose(last.values[-1], rolling.run(returns).values[-1])


def test_gaussian_on_a_last_calibration_gives_one_value_at_risk() -> None:
    vol = FittedEwmaVolatility(EwmaDecayCalibration(Last(length=100)))

    var = GaussianModel(cf=0.99, vol_model=vol).run(simulated_returns(n=160))

    assert np.isnan(var.values[:-1]).all()
    assert not np.isnan(var.values[-1]).any()
