import datetime

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from tail_lib.models.value_at_risk import Backcast
from tail_lib.models.value_at_risk import Cache
from tail_lib.models.value_at_risk import ConstantDecay
from tail_lib.models.value_at_risk import ConstantMean
from tail_lib.models.value_at_risk import ConstantVolatility
from tail_lib.models.value_at_risk import EwmaDecayCalibration
from tail_lib.models.value_at_risk import EwmaVolatility
from tail_lib.models.value_at_risk import FittedEwmaVolatility
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ModelPool
from tail_lib.models.value_at_risk import Preprocessor
from tail_lib.models.value_at_risk import StartingVol
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.value_at_risk import VolatilityFilter
from tail_lib.models.value_at_risk import VolatilityModel
from tail_lib.models.value_at_risk import fit_ewma_decay
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing

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


def _ewma(decay: np.ndarray | float, start: float = 0.01) -> EwmaVolatility:
    return EwmaVolatility(ConstantDecay(decay), start=StartingVol(start))


def test_ewma_volatility_recursion() -> None:
    returns = _returns(n=5)
    vols = _ewma(0.94).run(returns)

    # The volatility on date t is the forecast for t + 1: it uses r[t].
    expected = []
    previous = 0.01
    for r in returns.values:
        previous = np.sqrt(0.94 * previous**2 + 0.06 * r**2)
        expected.append(previous)

    np.testing.assert_array_equal(vols.dates, returns.dates)
    np.testing.assert_allclose(vols.values, expected)


def test_ewma_volatility_backcast_by_default() -> None:
    returns = _returns(n=30)
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
            _returns(n=30)
        )


@pytest.mark.parametrize("window", [0, -1])
def test_backcast_rejects_empty_window(window: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Backcast(window=window)


def test_starting_vol_rejects_non_positive_vol() -> None:
    with pytest.raises(ValueError, match="positive"):
        StartingVol(np.array([0.01, 0.0]))


def test_constant_decay_rejects_decay_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        ConstantDecay(1.1)


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
    # Kept apart: the two model types do not output the same type yet (D2).
    gaussians = {
        "gaussian_99": GaussianModel(cf=0.99, vol_model=vol_model),
        "gaussian_95": GaussianModel(cf=0.95, vol_model=vol_model),
    }
    hs = HistoricalSimulationsModel(cf=0.99)

    results = ModelPool({**gaussians, "hs_99": hs}).run(returns)

    assert CountingVolatility.n_calls == 1
    for name, model in gaussians.items():
        np.testing.assert_allclose(results[name].values, model.run(returns).values)
    np.testing.assert_allclose(results["hs_99"], hs.run(returns))


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
    ("bounds", "window"),
    [((0.9, 0.8), 20), ((0.5, 1.1), 20), ((0.8, 0.99), 300)],
)
def test_fit_ewma_decay_rejects_invalid_arguments(
    bounds: tuple[float, float], window: int
) -> None:
    r = _simulate_ewma(np.array([0.94]), n=300)

    with pytest.raises(ValueError, match=r"(bounds|Backcast)"):
        fit_ewma_decay(r, bounds=bounds, start=Backcast(window=window))


def test_fit_ewma_decay_with_starting_vol_uses_every_date() -> None:
    r = _simulate_ewma(np.array([0.9, 0.97]), n=300)

    fitted = fit_ewma_decay(r, start=StartingVol(0.01))

    assert fitted.shape == (2,)
    assert not np.allclose(fitted, fit_ewma_decay(r))


def test_ewma_volatility_with_batch_decay_factors() -> None:
    returns = _returns(n=50, n_assets=2)
    decay = np.array([0.9, 0.97])

    vols = _ewma(decay).run(returns)

    for i, d in enumerate(decay):
        single = TimeSeries(dates=returns.dates, values=returns.values[:, i])
        np.testing.assert_allclose(vols.values[:, i], _ewma(d).run(single).values)


def test_ewma_volatility_rejects_decay_not_matching_batch() -> None:
    returns = _returns(n=50, n_assets=2)
    model = _ewma(np.array([0.9, 0.94, 0.97]))

    with pytest.raises(ValueError, match="broadcast"):
        model.run(returns)


def _simulated_returns(n: int = 160) -> TimeSeries:
    return TimeSeries.from_first_date(
        FIRST_DATE, values=_simulate_ewma(np.array([0.9, 0.97]), n=n)
    )


def test_decay_calibration_fits_trailing_windows() -> None:
    returns = _simulated_returns()
    r = returns.values

    decay = EwmaDecayCalibration(Trailing(length=100, step=20)).run(returns)

    np.testing.assert_array_equal(decay.dates, returns.dates)
    assert np.isnan(decay.values[:99]).all()
    np.testing.assert_allclose(decay.values[99], fit_ewma_decay(r[:100]))
    np.testing.assert_allclose(decay.values[99:119], np.tile(decay.values[99], (20, 1)))
    np.testing.assert_allclose(decay.values[119], fit_ewma_decay(r[20:120]))
    np.testing.assert_allclose(decay.values[159], fit_ewma_decay(r[60:160]))


def test_decay_calibration_rejects_window_longer_than_returns() -> None:
    with pytest.raises(ValueError, match="longer"):
        EwmaDecayCalibration(Trailing(length=200)).run(_simulated_returns())


def test_ewma_volatility_compute_matches_constant_decay_run() -> None:
    returns = _returns(n=50, n_assets=2)
    decay = TimeSeries(dates=returns.dates, values=np.full((50, 2), 0.94))

    np.testing.assert_allclose(
        _ewma(0.94).compute(returns, decay=decay).values,
        _ewma(0.94).run(returns).values,
    )


def test_ewma_volatility_decay_applies_from_its_own_date() -> None:
    returns = _returns(n=50)
    switch = 30
    decay = TimeSeries(
        dates=returns.dates,
        values=np.where(np.arange(50) < switch, 0.9, 0.97),
    )

    vols = _ewma(0.9).compute(returns, decay=decay)
    constant = _ewma(0.9).run(returns)

    np.testing.assert_allclose(vols.values[:switch], constant.values[:switch])
    assert not np.isclose(vols.values[switch], constant.values[switch])


def test_ewma_volatility_rejects_decay_series_on_other_dates() -> None:
    returns = _returns(n=50)
    decay = TimeSeries.from_first_date(
        FIRST_DATE + datetime.timedelta(days=1), values=np.full(50, 0.94)
    )

    with pytest.raises(ValueError, match="dates"):
        _ewma(0.94).compute(returns, decay=decay)


def test_fitted_ewma_volatility_starts_after_first_calibration() -> None:
    returns = _simulated_returns()
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
    returns = _simulated_returns()
    vol_model = FittedEwmaVolatility(
        EwmaDecayCalibration(Trailing(length=100, step=20))
    )
    t = 130

    full = vol_model.run(returns)
    truncated = vol_model.run(returns[: t + 1])

    np.testing.assert_allclose(full.values[: t + 1], truncated.values)


def test_fitted_ewma_volatility_uses_calibrated_decay() -> None:
    returns = _simulated_returns()
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
    returns = _simulated_returns()
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
    returns = _simulated_returns()
    calibration = EwmaDecayCalibration(
        Trailing(length=100, step=20), start=StartingVol(0.01)
    )
    fit = calibration.run(returns).values[99]

    vols = FittedEwmaVolatility(calibration).run(returns)

    first_window = EwmaVolatility(ConstantDecay(fit), start=StartingVol(0.01))
    np.testing.assert_allclose(
        vols.values[99:119], first_window.run(returns[:119]).values[99:119]
    )


def test_pool_shares_decay_calibration_across_models() -> None:
    class _CountingCalibration(EwmaDecayCalibration):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> TimeSeries:
            _CountingCalibration.n_calls += 1
            return super().compute(returns)

    returns = _simulated_returns()
    calibration = _CountingCalibration(Trailing(length=100, step=20))

    # Two volatility models, each a different node, on one calibration.
    ModelPool(
        {
            "gaussian_99": GaussianModel(
                cf=0.99, vol_model=FittedEwmaVolatility(calibration)
            ),
            "gaussian_95": GaussianModel(
                cf=0.95, vol_model=FittedEwmaVolatility(calibration)
            ),
        }
    ).run(returns)

    assert _CountingCalibration.n_calls == 1


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

    def evaluate(self, cache: Cache) -> TimeSeries:
        filtered = cache.get(self._filter)
        vols = cache.get(self._vol_model)
        return self.compute(filtered=filtered, vols=vols)

    def compute(self, *, filtered: TimeSeries, vols: TimeSeries) -> TimeSeries:
        return TimeSeries(dates=filtered.dates, values=filtered.values * vols.values)


def test_volatility_filter_divides_returns_by_vols() -> None:
    returns = _returns(n=50, n_assets=2)
    vols = _ewma(0.94).run(returns)

    filtered = VolatilityFilter(_ewma(0.94)).compute(returns, vols=vols)

    # The return of date t is divided by the forecast for t, made on t - 1.
    assert np.isnan(filtered.values[0]).all()
    np.testing.assert_allclose(
        filtered.values[1:], returns.values[1:] / vols.values[:-1]
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
    np.testing.assert_allclose(rescaled.values[1:], returns.values[1:])


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

        def evaluate(self, cache: Cache) -> TimeSeries:
            if self.dependency is not None:
                cache.get(self.dependency)
            return cache.returns

    a, b = _Chained(), _Chained()
    a.dependency, b.dependency = b, a

    with pytest.raises(ValueError, match="depends on itself"):
        GaussianModel(cf=0.99, vol_model=a).run(_returns(n=50))
