import inspect
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.graph import Cache
from tail_lib.graph import Node
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.mean import ConstantMean
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import FittedEwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
from tail_lib.run import _input_name
from tail_lib.run import _window
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import simulated_returns


class _RecordingCache(Cache):
    """A cache that records every node asked for."""

    def __init__(
        self, returns: TimeSeries, inputs: Mapping[str, TimeSeries] | None = None
    ) -> None:
        super().__init__(returns, inputs)
        self.requested: list[Node[Any]] = []

    def get[T](self, node: Node[T]) -> T:
        self.requested.append(node)
        return super().get(node)


def _one_node_of_each_type() -> list[Node[Any]]:
    vol = EwmaVolatility(ConstantDecay(0.94))
    calibration = EwmaDecayCalibration(Trailing(length=100))
    vol_filter = VolatilityFilter(vol, mean_model=ConstantMean(1e-4))
    gaussian = GaussianModel(cf=0.99, vol_model=vol, mean_model=ConstantMean(1e-4))
    return [
        ZeroMean(),
        ConstantMean(1e-4),
        ConstantDecay(0.94),
        calibration,
        VolatilityInput("vol"),
        ConstantVolatility(0.01),
        vol,
        FittedEwmaVolatility(calibration),
        vol_filter,
        ValueAtRiskInput("var", cf=0.99),
        HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100)),
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=vol_filter, window=Trailing(length=100)
        ),
        gaussian,
        Overshootings(gaussian),
        KupiecTest(Overshootings(gaussian)),
    ]


def _library_node_types(cls: type[Node[Any]] = Node) -> Iterator[type[Node[Any]]]:
    for subclass in cls.__subclasses__():
        if subclass.__module__.startswith("tail_lib.") and not inspect.isabstract(
            subclass
        ):
            yield subclass
        yield from _library_node_types(subclass)


def test_every_node_type_is_checked() -> None:
    checked = {type(node) for node in _one_node_of_each_type()}

    assert checked == set(_library_node_types())


@pytest.mark.parametrize(
    "node", _one_node_of_each_type(), ids=lambda node: type(node).__name__
)
def test_dependencies_are_the_nodes_evaluate_reads(node: Node[Any]) -> None:
    returns = simulated_returns(n=300)
    run_input = TimeSeries(
        dates=returns.dates, values=np.full(returns.values.shape, -0.02)
    )
    cache = _RecordingCache(returns, inputs={"vol": run_input, "var": run_input})
    # Compute the dependencies first, so that only the nodes `evaluate` itself
    # asks for are recorded. An undeclared node would also record its own.
    dependencies = list(node.dependencies().values())
    for dependency in dependencies:
        cache.get(dependency)
    cache.requested.clear()

    node.evaluate(cache)

    assert set(cache.requested) == set(dependencies)


@pytest.mark.parametrize(
    "node", _one_node_of_each_type(), ids=lambda node: type(node).__name__
)
def test_run_sees_the_inputs_and_windows_of_every_node(node: Node[Any]) -> None:
    # A run finds its inputs and `last` windows by walking the graph: a node
    # with a `name` reads a run input, and one with a `window` is computed on
    # windows. The walk has to recognize each of them.
    assert _input_name(node) == getattr(node, "name", None)
    assert _window(node) is getattr(node, "window", None)
