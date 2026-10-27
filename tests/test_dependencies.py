import dataclasses
import inspect
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from tail_lib.backtests.kupiec import _KupiecTest
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.context import Context
from tail_lib.context import _input_name
from tail_lib.context import _window
from tail_lib.graph import Node
from tail_lib.graph import RunInput
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.mean import ConstantMean
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import simulated_returns


def _one_node_of_each_type() -> list[Node[Any]]:
    vol = EwmaVolatility(ConstantDecay(0.94))
    calibration = EwmaDecayCalibration(Trailing(length=100))
    vol_filter = VolatilityFilter(vol, mean=ConstantMean(1e-4))
    gaussian = GaussianModel(cf=0.99, vol=vol, mean=ConstantMean(1e-4))
    return [
        ZeroMean(),
        ConstantMean(1e-4),
        ConstantDecay(0.94),
        calibration,
        VolatilityInput("vol"),
        ConstantVolatility(0.01),
        vol,
        EwmaVolatility(calibration),
        vol_filter,
        ValueAtRiskInput("var", cf=0.99),
        HistoricalSimulationsModel(cf=0.99, window=Trailing(length=100)),
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=vol_filter, window=Trailing(length=100)
        ),
        gaussian,
        Overshootings(gaussian),
        _KupiecTest(Overshootings(gaussian)),
    ]


# Base classes that aren't abstract, but aren't node types a run holds either.
_BASE_CLASSES = {RunInput}


def _library_node_types(cls: type[Node[Any]] = Node) -> Iterator[type[Node[Any]]]:
    for subclass in cls.__subclasses__():
        if (
            subclass.__module__.startswith("tail_lib.")
            and not inspect.isabstract(subclass)
            and subclass not in _BASE_CLASSES
        ):
            yield subclass
        yield from _library_node_types(subclass)


def test_every_node_type_is_checked() -> None:
    checked = {type(node) for node in _one_node_of_each_type()}

    assert checked == set(_library_node_types())


@pytest.mark.parametrize(
    "node", _one_node_of_each_type(), ids=lambda node: type(node).__name__
)
def test_context_passes_each_declared_inputs_output_to_compute(
    node: Node[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    returns = simulated_returns(n=300)
    run_input = TimeSeries(
        dates=returns.dates, values=np.full(returns.values.shape, -0.02)
    )
    bound = (
        dataclasses.replace(node, name="model")
        if isinstance(node, ValueAtRiskModel) and node.name is None
        else node
    )
    ctx = Context()
    ctx.bind(bound)
    # Outputs differ by node type. Record their identities at the compute
    # boundary, after node construction has checked its real signature (D11).
    calls: list[dict[str, object]] = []
    compute = type(node).compute

    def recording(self: Node[Any], returns: TimeSeries, /, **inputs: object) -> object:
        calls.append(inputs)
        return compute(self, returns, **inputs)

    monkeypatch.setattr(type(node), "compute", recording)
    results = ctx.run(returns, {"vol": run_input, "var": run_input})

    if isinstance(node, RunInput):
        assert not calls
        assert results[node] is run_input
    else:
        assert len(calls) == 1
        assert calls[0].keys() == node.dependencies().keys()
        for name, dependency in node.dependencies().items():
            assert calls[0][name] is results[dependency]


@pytest.mark.parametrize(
    "node", _one_node_of_each_type(), ids=lambda node: type(node).__name__
)
def test_run_sees_the_inputs_and_windows_of_every_node(node: Node[Any]) -> None:
    # A run finds its inputs and `last` windows by walking the graph: a node
    # with a `name` reads a run input, and one with a `window` is computed on
    # windows. The walk has to recognize each of them.
    assert _input_name(node) == getattr(node, "name", None)
    assert _window(node) is getattr(node, "window", None)


@pytest.mark.parametrize(
    "node", _one_node_of_each_type(), ids=lambda node: type(node).__name__
)
def test_settings_and_dependencies_cover_the_constructor_arguments(
    node: Node[Any],
) -> None:
    # Every constructor argument is either a setting or a dependency, and
    # nothing else is, but a node's `name`, which isn't part of its identity
    # (a run input's is, and is a setting).
    # A node without its own `__init__` shows `object`'s `*args, **kwargs`.
    parameters = inspect.signature(type(node).__init__).parameters.values()
    arguments = {
        parameter.name
        for parameter in parameters
        if parameter.name != "self"
        and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    }
    if not isinstance(node, RunInput):
        arguments.remove("name")
    settings = set(node.settings())
    dependencies = set(node.dependencies())

    assert not settings & dependencies
    assert settings | dependencies == arguments


@pytest.mark.parametrize(
    "node", _one_node_of_each_type(), ids=lambda node: type(node).__name__
)
def test_building_a_node_checks_its_compute(
    node: Node[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every `__post_init__` has to reach `Node.__post_init__`, which checks
    # `compute` against the inputs (D11).
    checked: list[Node[Any]] = []
    check = Node._check_compute  # noqa: SLF001

    def recording(self: Node[Any]) -> None:
        checked.append(self)
        check(self)

    monkeypatch.setattr(Node, "_check_compute", recording)
    copy = dataclasses.replace(node)

    assert checked == [copy]
