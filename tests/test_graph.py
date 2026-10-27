from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from tail_lib.graph import Cache
from tail_lib.graph import Dependencies
from tail_lib.graph import Pool
from tail_lib.graph import Preprocessor
from tail_lib.graph import Settings
from tail_lib.models.ewma import DecayFactors
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import FilteredReturns
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
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


@dataclass(frozen=True, kw_only=True)
class _FilteredReturnsModel(ValueAtRiskModel):
    """Depends on a volatility model both directly and through a filter."""

    filter: VolatilityFilter
    vol: VolatilityModel

    def compute(
        self,
        returns: TimeSeries,  # noqa: ARG002
        /,
        *,
        filter: FilteredReturns,  # noqa: A002
        vol: TimeSeries,
    ) -> TimeSeries:
        z = filter.standardized
        return TimeSeries(dates=z.dates, values=z.values * vol.values)


def test_run_computes_nested_preprocessors_once() -> None:
    returns = returns_series(n=50)
    vol = _CountingVolatility(0.02)

    model = _FilteredReturnsModel(cf=0.99, filter=VolatilityFilter(vol), vol=vol)

    rescaled = model.run(returns)

    assert vol.n_calls == 1
    np.testing.assert_allclose(rescaled.values[1:], returns.values[1:])


def test_run_rejects_cyclic_preprocessors() -> None:
    # Frozen nodes can't be built into a cycle: this one sets a plain
    # attribute after it is built, which a subclass that isn't a dataclass can.
    class _Chained(VolatilityModel):
        def __init__(self) -> None:
            self.dependency: Preprocessor | None = None

        def dependencies(self) -> Dependencies:
            return {} if self.dependency is None else {"dependency": self.dependency}

        def settings(self) -> Settings:
            return {}

        # Never reached: the cache finds the cycle while getting the inputs.
        def compute(
            self,
            returns: TimeSeries,
            /,
            *,
            dependency: TimeSeries | None = None,  # noqa: ARG002
        ) -> TimeSeries:
            return returns

    a, b = _Chained(), _Chained()
    a.dependency, b.dependency = b, a

    with pytest.raises(ValueError, match="depends on itself"):
        GaussianModel(cf=0.99, vol=a).run(returns_series(n=50))


def test_pool_computes_shared_preprocessors_once() -> None:
    class CountingVolatility(ConstantVolatility):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> TimeSeries:
            CountingVolatility.n_calls += 1
            return super().compute(returns)

    returns = returns_series()
    vol = CountingVolatility(0.01)
    models = {
        "gaussian_99": GaussianModel(cf=0.99, vol=vol),
        "gaussian_95": GaussianModel(cf=0.95, vol=vol),
        "hs_99": HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100)),
    }

    results = Pool(models).run(returns)

    assert CountingVolatility.n_calls == 1
    for name, model in models.items():
        np.testing.assert_allclose(results[name].values, model.run(returns).values)


def test_pool_shares_nested_preprocessors_across_models() -> None:
    returns = returns_series(n=50)
    vol = _CountingVolatility(0.02)

    Pool(
        {
            "filtered": _FilteredReturnsModel(
                cf=0.99, filter=VolatilityFilter(vol), vol=vol
            ),
            "gaussian": GaussianModel(cf=0.99, vol=vol),
        }
    ).run(returns)

    assert vol.n_calls == 1


def test_pool_shares_decay_calibration_across_models() -> None:
    class _CountingCalibration(EwmaDecayCalibration):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> DecayFactors:
            _CountingCalibration.n_calls += 1
            return super().compute(returns)

    returns = simulated_returns()
    calibration = _CountingCalibration(Trailing(length=100, step=20))

    # Two volatility models, each a different node, on one calibration.
    Pool(
        {
            "gaussian_99": GaussianModel(cf=0.99, vol=EwmaVolatility(calibration)),
            "gaussian_95": GaussianModel(cf=0.95, vol=EwmaVolatility(calibration)),
        }
    ).run(returns)

    assert _CountingCalibration.n_calls == 1


def test_pools_on_one_cache_share_nodes() -> None:
    returns = returns_series(n=50)
    vol = _CountingVolatility(0.02)
    gaussian = GaussianModel(cf=0.99, vol=vol)
    cache = Cache(returns)

    first = Pool({"gaussian": gaussian}).evaluate(cache)
    second = Pool({"vols": vol, "gaussian": gaussian}).evaluate(cache)

    assert vol.n_calls == 1
    assert second["gaussian"] is first["gaussian"]


def test_var_model_output_is_cached() -> None:
    cache = Cache(returns_series(n=50))
    gaussian = GaussianModel(cf=0.99, vol=ConstantVolatility(0.02))

    assert cache.get(gaussian) is cache.get(gaussian)


@dataclass(frozen=True)
class _Scaled(Preprocessor):
    """Scales its input's output: `compute` must take `vol`, its one input."""

    vol: VolatilityModel
    factor: float = 2.0

    def compute(self, returns: TimeSeries, /, *, vol: TimeSeries) -> TimeSeries:
        return TimeSeries(dates=returns.dates, values=self.factor * vol.values)


def test_compute_gets_the_output_of_each_input_by_field_name() -> None:
    returns = returns_series(n=10)

    scaled = _Scaled(ConstantVolatility(0.01)).run(returns)

    np.testing.assert_allclose(scaled.values, 0.02)


# Nodes whose `compute` doesn't match their inputs. Their bodies never run.


@dataclass(frozen=True)
class _MisnamedInput(Preprocessor):
    volatility: VolatilityModel

    def compute(self, returns: TimeSeries, /, *, vol: TimeSeries) -> TimeSeries:  # noqa: ARG002
        return vol


@dataclass(frozen=True)
class _ExtraKeyword(Preprocessor):
    vol: VolatilityModel

    def compute(
        self,
        returns: TimeSeries,  # noqa: ARG002
        /,
        *,
        vol: TimeSeries,
        mean: TimeSeries,  # noqa: ARG002
    ) -> TimeSeries:
        return vol


@dataclass(frozen=True)
class _AnyKeywords(Preprocessor):
    vol: VolatilityModel

    def compute(self, returns: TimeSeries, /, **inputs: TimeSeries) -> TimeSeries:  # noqa: ARG002
        return inputs["vol"]


@dataclass(frozen=True)
class _NoReturns(Preprocessor):
    vol: VolatilityModel

    def compute(self, *, vol: TimeSeries) -> TimeSeries:
        return vol


@pytest.mark.parametrize(
    ("node_type", "match"),
    [
        (_MisnamedInput, r"Missing: \['volatility'\]. Not input fields: \['vol'\]"),
        (_ExtraKeyword, r"Missing: \[\]. Not input fields: \['mean'\]"),
        (_AnyKeywords, "without \\*args or \\*\\*kwargs"),
        (_NoReturns, "has to take the returns"),
    ],
)
def test_a_compute_not_matching_the_inputs_fails_when_the_node_is_built(
    node_type: Callable[[VolatilityModel], Preprocessor], match: str
) -> None:
    with pytest.raises(TypeError, match=match):
        node_type(ConstantVolatility(0.01))


def test_a_name_is_not_part_of_a_nodes_identity() -> None:
    named = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
    renamed = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="other")
    unnamed = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))

    assert named == renamed == unnamed
    assert hash(named) == hash(renamed) == hash(unnamed)
    assert named.name == "gauss"
    assert unnamed.name is None
    assert "name" not in named.settings()
    assert "gauss" not in repr(named)


def test_equal_nodes_with_different_names_are_computed_once() -> None:
    vol = _CountingVolatility(0.02)

    Pool(
        {
            "a": GaussianModel(cf=0.99, vol=vol, name="a"),
            "b": GaussianModel(cf=0.99, vol=vol, name="b"),
        }
    ).run(returns_series(n=50))

    assert vol.n_calls == 1


@pytest.mark.parametrize("name", ["gauss-99", "1gauss", "gauss 99", ""])
def test_a_node_name_has_to_be_an_identifier(name: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        ConstantVolatility(0.01, name=name)


def test_a_run_inputs_name_is_part_of_its_identity() -> None:
    assert VolatilityInput("a") == VolatilityInput("a")
    assert VolatilityInput("a") != VolatilityInput("b")
    assert ValueAtRiskInput("a", cf=0.99).name == "a"
    assert VolatilityInput("a").settings() == {"name": "a"}
