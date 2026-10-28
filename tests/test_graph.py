from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from tail_lib.context import Context
from tail_lib.graph import Node
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
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Trailing
from tests.helpers import returns_series
from tests.helpers import run_node
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

    rescaled = run_node(model, returns)

    assert vol.n_calls == 1
    np.testing.assert_allclose(rescaled.values[1:], returns.values[1:])


def test_context_computes_shared_preprocessors_once() -> None:
    class CountingVolatility(ConstantVolatility):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> TimeSeries:
            CountingVolatility.n_calls += 1
            return super().compute(returns)

    returns = returns_series()
    vol = CountingVolatility(0.01)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=vol, name="gaussian_99")
        GaussianModel(cf=0.95, vol=vol, name="gaussian_95")
        HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100), name="hs_99")

    results = ctx.run(returns)

    assert CountingVolatility.n_calls == 1
    for name, model in ctx.models.items():
        np.testing.assert_allclose(
            results[name].values, run_node(model, returns).values
        )


def test_context_shares_nested_preprocessors_across_models() -> None:
    returns = returns_series(n=50)
    vol = _CountingVolatility(0.02)

    with Context() as ctx:
        _FilteredReturnsModel(
            cf=0.99, filter=VolatilityFilter(vol), vol=vol, name="filtered"
        )
        GaussianModel(cf=0.99, vol=vol, name="gaussian")

    ctx.run(returns)

    assert vol.n_calls == 1


def test_context_shares_decay_calibration_across_models() -> None:
    class _CountingCalibration(EwmaDecayCalibration):
        n_calls = 0

        def compute(self, returns: TimeSeries) -> DecayFactors:
            _CountingCalibration.n_calls += 1
            return super().compute(returns)

    returns = simulated_returns()
    calibration = _CountingCalibration(Trailing(length=100, step=20))

    # Two VaR models on equal volatility nodes and one calibration.
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=EwmaVolatility(calibration), name="gaussian_99")
        GaussianModel(cf=0.95, vol=EwmaVolatility(calibration), name="gaussian_95")

    ctx.run(returns)

    assert _CountingCalibration.n_calls == 1


@dataclass(frozen=True)
class _Scaled(Node[TimeSeries]):
    """Scales its input's output: `compute` must take `vol`, its one input."""

    vol: VolatilityModel
    factor: float = 2.0

    def compute(self, returns: TimeSeries, /, *, vol: TimeSeries) -> TimeSeries:
        return TimeSeries(dates=returns.dates, values=self.factor * vol.values)


def test_compute_gets_the_output_of_each_input_by_field_name() -> None:
    returns = returns_series(n=10)

    scaled = run_node(_Scaled(ConstantVolatility(0.01)), returns)

    np.testing.assert_allclose(scaled.values, 0.02)


# Nodes whose `compute` doesn't match their inputs. Their bodies never run.


@dataclass(frozen=True)
class _MisnamedInput(Node[TimeSeries]):
    volatility: VolatilityModel

    def compute(self, returns: TimeSeries, /, *, vol: TimeSeries) -> TimeSeries:  # noqa: ARG002
        return vol


@dataclass(frozen=True)
class _ExtraKeyword(Node[TimeSeries]):
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
class _AnyKeywords(Node[TimeSeries]):
    vol: VolatilityModel

    def compute(self, returns: TimeSeries, /, **inputs: TimeSeries) -> TimeSeries:  # noqa: ARG002
        return inputs["vol"]


@dataclass(frozen=True)
class _NoReturns(Node[TimeSeries]):
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
    node_type: Callable[[VolatilityModel], Node[TimeSeries]], match: str
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

    with Context() as ctx:
        GaussianModel(cf=0.99, vol=vol, name="a")
        GaussianModel(cf=0.99, vol=vol, name="b")

    ctx.run(returns_series(n=50))

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
