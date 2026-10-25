import numpy as np
import pytest

from tail_lib.graph import Cache
from tail_lib.graph import Dependencies
from tail_lib.graph import Pool
from tail_lib.graph import Preprocessor
from tail_lib.graph import Settings
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import FittedEwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityModel
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import returns_series
from tests.helpers import simulated_returns


class _CountingVolatility(ConstantVolatility):
    def __init__(self, vol: float) -> None:
        super().__init__(vol)
        self.n_calls = 0

    def compute(self, returns: TimeSeries) -> TimeSeries:
        self.n_calls += 1
        return super().compute(returns)


class _FilteredReturnsModel(ValueAtRiskModel):
    """Depends on a volatility model both directly and through a filter."""

    def __init__(self, *, cf: float, vol_model: VolatilityModel) -> None:
        super().__init__(cf=cf)
        self._vol_model = vol_model
        self._filter = VolatilityFilter(vol_model)

    def dependencies(self) -> Dependencies:
        return {"filter": self._filter, "vol_model": self._vol_model}

    def settings(self) -> Settings:
        return {"cf": self._cf}

    def evaluate(self, cache: Cache) -> TimeSeries:
        filtered = cache.get(self._filter)
        vols = cache.get(self._vol_model)
        return self.compute(filtered=filtered, vols=vols)

    def compute(self, *, filtered: TimeSeries, vols: TimeSeries) -> TimeSeries:
        return TimeSeries(dates=filtered.dates, values=filtered.values * vols.values)


def test_run_computes_nested_preprocessors_once() -> None:
    returns = returns_series(n=50)
    vol_model = _CountingVolatility(0.02)

    rescaled = _FilteredReturnsModel(cf=0.99, vol_model=vol_model).run(returns)

    assert vol_model.n_calls == 1
    np.testing.assert_allclose(rescaled.values[1:], returns.values[1:])


def test_run_rejects_cyclic_preprocessors() -> None:
    class _Chained(VolatilityModel):
        def __init__(self) -> None:
            self.dependency: Preprocessor | None = None

        def dependencies(self) -> Dependencies:
            return {} if self.dependency is None else {"dependency": self.dependency}

        def settings(self) -> Settings:
            return {}

        def evaluate(self, cache: Cache) -> TimeSeries:
            if self.dependency is not None:
                cache.get(self.dependency)
            return cache.returns

    a, b = _Chained(), _Chained()
    a.dependency, b.dependency = b, a

    with pytest.raises(ValueError, match="depends on itself"):
        GaussianModel(cf=0.99, vol_model=a).run(returns_series(n=50))


def test_pool_computes_shared_preprocessors_once() -> None:
    class CountingVolatility(ConstantVolatility):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> TimeSeries:
            CountingVolatility.n_calls += 1
            return super().compute(returns)

    returns = returns_series()
    vol_model = CountingVolatility(0.01)
    models = {
        "gaussian_99": GaussianModel(cf=0.99, vol_model=vol_model),
        "gaussian_95": GaussianModel(cf=0.95, vol_model=vol_model),
        "hs_99": HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100)),
    }

    results = Pool(models).run(returns)

    assert CountingVolatility.n_calls == 1
    for name, model in models.items():
        np.testing.assert_allclose(results[name].values, model.run(returns).values)


def test_pool_shares_nested_preprocessors_across_models() -> None:
    returns = returns_series(n=50)
    vol_model = _CountingVolatility(0.02)

    Pool(
        {
            "filtered": _FilteredReturnsModel(cf=0.99, vol_model=vol_model),
            "gaussian": GaussianModel(cf=0.99, vol_model=vol_model),
        }
    ).run(returns)

    assert vol_model.n_calls == 1


def test_pool_shares_decay_calibration_across_models() -> None:
    class _CountingCalibration(EwmaDecayCalibration):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> TimeSeries:
            _CountingCalibration.n_calls += 1
            return super().compute(returns)

    returns = simulated_returns()
    calibration = _CountingCalibration(Trailing(length=100, step=20))

    # Two volatility models, each a different node, on one calibration.
    Pool(
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


def test_pools_on_one_cache_share_nodes() -> None:
    returns = returns_series(n=50)
    vol_model = _CountingVolatility(0.02)
    gaussian = GaussianModel(cf=0.99, vol_model=vol_model)
    cache = Cache(returns)

    first = Pool({"gaussian": gaussian}).evaluate(cache)
    second = Pool({"vols": vol_model, "gaussian": gaussian}).evaluate(cache)

    assert vol_model.n_calls == 1
    assert second["gaussian"] is first["gaussian"]


def test_var_model_output_is_cached() -> None:
    cache = Cache(returns_series(n=50))
    gaussian = GaussianModel(cf=0.99, vol_model=ConstantVolatility(0.02))

    assert cache.get(gaussian) is cache.get(gaussian)
