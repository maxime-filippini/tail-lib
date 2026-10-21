import datetime
from collections.abc import Mapping

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from tail_lib.models.value_at_risk import ConstantMean
from tail_lib.models.value_at_risk import ConstantVolatility
from tail_lib.models.value_at_risk import EwmaDecayCalibration
from tail_lib.models.value_at_risk import EwmaVolatility
from tail_lib.models.value_at_risk import FittedEwmaVolatility
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ModelPool
from tail_lib.models.value_at_risk import Preprocessor
from tail_lib.models.value_at_risk import Rolling
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.value_at_risk import VolatilityFilter
from tail_lib.models.value_at_risk import VolatilityModel
from tail_lib.models.value_at_risk import fit_ewma_decay
from tail_lib.time_series import TimeSeries

FIRST_DATE = datetime.date(2020, 1, 1)


def _returns(n: int = 250, n_assets: int | None = None) -> TimeSeries:
    rng = np.random.default_rng(0)
    shape = (n,) if n_assets is None else (n, n_assets)
    return TimeSeries.from_first_date(
        FIRST_DATE, values=rng.normal(scale=0.01, size=shape)
    )


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
    returns = _returns(n_assets=2)
    model = GaussianModel(
        cf=0.99,
        mean_model=ConstantMean(np.array([0.001, 0.002])),
        vol_model=ConstantVolatility(np.array([0.01, 0.02])),
    )

    var = model.run(returns)

    np.testing.assert_array_equal(var.dates, returns.dates)
    expected = stats.norm.ppf(0.01, loc=[0.001, 0.002], scale=[0.01, 0.02])
    np.testing.assert_allclose(var.values, np.tile(expected, (returns.n_dates, 1)))


def test_ewma_volatility_recursion() -> None:
    returns = _returns(n=5)
    vols = EwmaVolatility(decay_factor=0.94, starting_vol=0.01).compute(returns)

    expected = [0.01]
    for r in returns.values[:-1]:
        expected.append(np.sqrt(0.94 * expected[-1] ** 2 + 0.06 * r**2))

    np.testing.assert_array_equal(vols.dates, returns.dates)
    np.testing.assert_allclose(vols.values, expected)


def test_historical_simulations_run() -> None:
    returns = _returns(n_assets=2)

    var = HistoricalSimulationsModel(cf=0.99).run(returns)

    np.testing.assert_allclose(var, np.quantile(returns.values, 0.01, axis=0))


def test_pool_computes_shared_preprocessors_once() -> None:
    class CountingVolatility(ConstantVolatility):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> TimeSeries:
            CountingVolatility.n_calls += 1
            return super().compute(returns)

    returns = _returns()
    vol_model = CountingVolatility(0.01)
    models = {
        "gaussian_99": GaussianModel(cf=0.99, vol_model=vol_model),
        "gaussian_95": GaussianModel(cf=0.95, vol_model=vol_model),
        "hs_99": HistoricalSimulationsModel(cf=0.99),
    }

    results = ModelPool(models).run(returns)

    assert CountingVolatility.n_calls == 1
    for name in ("gaussian_99", "gaussian_95"):
        np.testing.assert_allclose(
            results[name].values, models[name].run(returns).values
        )
    np.testing.assert_allclose(results["hs_99"], models["hs_99"].run(returns))


def _simulate_ewma(decay_factor: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    var = np.full(decay_factor.shape, 1e-4)
    r = np.empty((n, *decay_factor.shape))
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal(decay_factor.shape)
        var = decay_factor * var + (1 - decay_factor) * r[t] ** 2
    return r


def test_fit_ewma_decay_recovers_decay_per_batch_element() -> None:
    true = np.array([[0.9, 0.94, 0.97], [0.92, 0.96, 0.98]])
    r = _simulate_ewma(true, n=2000)

    fitted = fit_ewma_decay(r)

    assert fitted.shape == true.shape
    np.testing.assert_allclose(fitted, true, atol=0.02)


def test_fit_ewma_decay_batched_matches_individual_fits() -> None:
    r = _simulate_ewma(np.array([0.9, 0.97]), n=300)

    fitted = fit_ewma_decay(r)

    for i in range(r.shape[1]):
        np.testing.assert_allclose(fitted[i], fit_ewma_decay(r[:, i]))


def test_fit_ewma_decay_on_rolling_windows() -> None:
    r = _simulate_ewma(np.array([0.9, 0.97]), n=300)
    w = 250
    windows = np.moveaxis(sliding_window_view(r, w, axis=0), -1, 0)

    fitted = fit_ewma_decay(windows)

    assert fitted.shape == (r.shape[0] - w + 1, r.shape[1])
    np.testing.assert_allclose(fitted[-1], fit_ewma_decay(r[-w:]))


@pytest.mark.parametrize(
    ("bounds", "burn_in"),
    [((0.9, 0.8), 20), ((0.5, 1.1), 20), ((0.8, 0.99), 0), ((0.8, 0.99), 300)],
)
def test_fit_ewma_decay_rejects_invalid_arguments(
    bounds: tuple[float, float], burn_in: int
) -> None:
    r = _simulate_ewma(np.array([0.94]), n=300)

    with pytest.raises(ValueError, match=r"(bounds|Burn-in)"):
        fit_ewma_decay(r, bounds=bounds, burn_in=burn_in)


def test_ewma_volatility_with_batch_decay_factors() -> None:
    returns = _returns(n=50, n_assets=2)
    decay = np.array([0.9, 0.97])

    vols = EwmaVolatility(decay_factor=decay, starting_vol=0.01).compute(returns)

    for i, d in enumerate(decay):
        single = TimeSeries(dates=returns.dates, values=returns.values[:, i])
        np.testing.assert_allclose(
            vols.values[:, i],
            EwmaVolatility(decay_factor=d, starting_vol=0.01).compute(single).values,
        )


def test_ewma_volatility_rejects_decay_not_matching_batch() -> None:
    returns = _returns(n=50, n_assets=2)
    model = EwmaVolatility(decay_factor=np.array([0.9, 0.94, 0.97]), starting_vol=0.01)

    with pytest.raises(ValueError, match="does not broadcast"):
        model.compute(returns)


def _simulated_returns(n: int = 160) -> TimeSeries:
    return TimeSeries.from_first_date(
        FIRST_DATE, values=_simulate_ewma(np.array([0.9, 0.97]), n=n)
    )


def test_rolling_schedule_calibrates_on_trailing_windows() -> None:
    returns = _simulated_returns()
    r = returns.values

    decay = Rolling(window=100, refit_every=20).calibrate(returns, fit=fit_ewma_decay)

    np.testing.assert_array_equal(decay.dates, returns.dates)
    assert np.isnan(decay.values[:99]).all()
    np.testing.assert_allclose(decay.values[99], fit_ewma_decay(r[:100]))
    np.testing.assert_allclose(decay.values[99:119], np.tile(decay.values[99], (20, 1)))
    np.testing.assert_allclose(decay.values[119], fit_ewma_decay(r[20:120]))
    np.testing.assert_allclose(decay.values[159], fit_ewma_decay(r[60:160]))


def test_rolling_schedule_rejects_window_longer_than_returns() -> None:
    with pytest.raises(ValueError, match="longer"):
        Rolling(window=200).calibrate(_simulated_returns(), fit=fit_ewma_decay)


def test_ewma_volatility_constant_decay_series_matches_constant_decay() -> None:
    returns = _returns(n=50, n_assets=2)
    decay = TimeSeries(dates=returns.dates, values=np.full((50, 2), 0.94))

    np.testing.assert_allclose(
        EwmaVolatility(decay_factor=decay, starting_vol=0.01).compute(returns).values,
        EwmaVolatility(decay_factor=0.94, starting_vol=0.01).compute(returns).values,
    )


def test_ewma_volatility_decay_series_applies_from_next_date() -> None:
    returns = _returns(n=50)
    switch = 30
    decay = TimeSeries(
        dates=returns.dates,
        values=np.where(np.arange(50) < switch, 0.9, 0.97),
    )

    vols = EwmaVolatility(decay_factor=decay, starting_vol=0.01).compute(returns)
    constant = EwmaVolatility(decay_factor=0.9, starting_vol=0.01).compute(returns)

    np.testing.assert_allclose(vols.values[: switch + 1], constant.values[: switch + 1])
    assert not np.isclose(vols.values[switch + 1], constant.values[switch + 1])


def test_ewma_volatility_rejects_decay_series_on_other_dates() -> None:
    returns = _returns(n=50)
    decay = TimeSeries.from_first_date(
        FIRST_DATE + datetime.timedelta(days=1), values=np.full(50, 0.94)
    )

    with pytest.raises(ValueError, match="dates"):
        EwmaVolatility(decay_factor=decay, starting_vol=0.01).compute(returns)


def test_fitted_ewma_volatility_starts_after_first_calibration() -> None:
    returns = _simulated_returns()
    vol_model = FittedEwmaVolatility(
        EwmaDecayCalibration(Rolling(window=100, refit_every=20))
    )

    vols = vol_model.run(returns)
    var = GaussianModel(cf=0.99, vol_model=vol_model).run(returns)

    np.testing.assert_array_equal(vols.dates, returns.dates)
    assert np.isnan(vols.values[:100]).all()
    assert np.isfinite(vols.values[100:]).all()
    assert np.isnan(var.values[:100]).all()
    assert np.isfinite(var.values[100:]).all()


def test_fitted_ewma_volatility_has_no_look_ahead() -> None:
    returns = _simulated_returns()
    vol_model = FittedEwmaVolatility(
        EwmaDecayCalibration(Rolling(window=100, refit_every=20))
    )
    t = 130

    full = vol_model.run(returns)
    truncated = vol_model.run(returns[: t + 1])

    np.testing.assert_allclose(full.values[: t + 1], truncated.values)


def test_fitted_ewma_volatility_uses_calibrated_decay() -> None:
    returns = _simulated_returns()
    calibration = EwmaDecayCalibration(Rolling(window=100, refit_every=20))
    vol_model = FittedEwmaVolatility(calibration)

    decay = calibration.run(returns)

    np.testing.assert_allclose(
        vol_model.run(returns).values,
        vol_model.compute(returns, decay=decay).values,
    )


def test_pool_shares_decay_calibration_across_models() -> None:
    class _CountingRolling(Rolling):
        n_calls = 0

        def calibrate(self, *args, **kwargs) -> TimeSeries:  # noqa: ANN002, ANN003
            _CountingRolling.n_calls += 1
            return super().calibrate(*args, **kwargs)

    returns = _simulated_returns()
    calibration = EwmaDecayCalibration(_CountingRolling(window=100, refit_every=20))

    ModelPool(
        {
            "default_start": GaussianModel(
                cf=0.99, vol_model=FittedEwmaVolatility(calibration)
            ),
            "custom_start": GaussianModel(
                cf=0.99,
                vol_model=FittedEwmaVolatility(calibration, starting_vol=0.02),
            ),
        }
    ).run(returns)

    assert _CountingRolling.n_calls == 1


class _CountingVolatility(ConstantVolatility):
    def __init__(self, vol: float) -> None:
        super().__init__(vol)
        self.n_calls = 0

    def compute(self, returns: TimeSeries) -> TimeSeries:
        self.n_calls += 1
        return super().compute(returns)


class _FilteredReturnsModel(ValueAtRiskModel[TimeSeries]):
    """Depends on a volatility model both directly and through a filter."""

    def __init__(self, *, cf: float, vol_model: VolatilityModel) -> None:
        super().__init__(cf=cf)
        self._vol_model = vol_model
        self._filter = VolatilityFilter(vol_model)

    @property
    def preprocessors(self) -> Mapping[str, Preprocessor]:
        return {"filtered": self._filter, "vols": self._vol_model}

    def compute(self, *, filtered: TimeSeries, vols: TimeSeries) -> TimeSeries:
        return TimeSeries(dates=filtered.dates, values=filtered.values * vols.values)


def test_volatility_filter_divides_returns_by_vols() -> None:
    returns = _returns(n=50, n_assets=2)
    vol_model = EwmaVolatility(decay_factor=0.94, starting_vol=0.01)

    filtered = VolatilityFilter(vol_model).compute(
        returns, vols=vol_model.compute(returns)
    )

    np.testing.assert_allclose(
        filtered.values, returns.values / vol_model.compute(returns).values
    )


def test_volatility_filter_rejects_mismatched_vols() -> None:
    returns = _returns(n=50)
    vols = TimeSeries.from_first_date(
        FIRST_DATE + datetime.timedelta(days=1), values=np.full(50, 0.01)
    )

    with pytest.raises(ValueError, match="dates"):
        VolatilityFilter(ConstantVolatility(0.01)).compute(returns, vols=vols)


def test_run_computes_nested_preprocessors_once() -> None:
    returns = _returns(n=50)
    vol_model = _CountingVolatility(0.02)

    rescaled = _FilteredReturnsModel(cf=0.99, vol_model=vol_model).run(returns)

    assert vol_model.n_calls == 1
    np.testing.assert_allclose(rescaled.values, returns.values)


def test_pool_shares_nested_preprocessors_across_models() -> None:
    returns = _returns(n=50)
    vol_model = _CountingVolatility(0.02)

    ModelPool(
        {
            "filtered": _FilteredReturnsModel(cf=0.99, vol_model=vol_model),
            "gaussian": GaussianModel(cf=0.99, vol_model=vol_model),
        }
    ).run(returns)

    assert vol_model.n_calls == 1


def test_run_rejects_cyclic_preprocessors() -> None:
    class _Chained(VolatilityModel):
        def __init__(self) -> None:
            self.dependency: Preprocessor | None = None

        @property
        def preprocessors(self) -> Mapping[str, Preprocessor]:
            return {} if self.dependency is None else {"vols": self.dependency}

        def compute(self, returns: TimeSeries, **_: TimeSeries) -> TimeSeries:
            return returns

    a, b = _Chained(), _Chained()
    a.dependency, b.dependency = b, a

    with pytest.raises(ValueError, match="depends on itself"):
        GaussianModel(cf=0.99, vol_model=a).run(_returns(n=50))
