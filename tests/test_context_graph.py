import dataclasses
from typing import Any

import pytest

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.context import Context
from tail_lib.context import GraphEdge
from tail_lib.context import RunGraph
from tail_lib.graph import Node
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import returns_series


class _CountingVolatility(ConstantVolatility):
    def __init__(self, vol: float) -> None:
        super().__init__(vol)
        self.n_calls = 0

    def compute(self, returns: TimeSeries) -> TimeSeries:
        self.n_calls += 1
        return super().compute(returns)


def _paths(graph: RunGraph) -> dict[Node[Any], tuple[str, ...]]:
    return {graph_node.node: graph_node.paths for graph_node in graph.nodes}


def _gauss_and_fhs() -> tuple[Context, ConstantVolatility, GaussianModel]:
    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        vol = ConstantVolatility(0.01)
        gauss = GaussianModel(cf=0.99, vol=vol, name="gauss")
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=VolatilityFilter(vol), window=Trailing(50), name="fhs"
        )
    return ctx, vol, gauss


def test_the_graph_lists_each_node_once_with_all_its_paths() -> None:
    run, vol, gauss = _gauss_and_fhs()

    paths = _paths(run.graph())

    # Each distinct node once: 2 models, the filter, the shared volatility,
    # one zero mean (the Gaussian's and the filter's are equal, so one node,
    # D9), then per model one overshootings node and one backtest.
    n_distinct = 2 + 1 + 1 + 1 + 2 * 2
    assert len(run.graph().nodes) == len(paths) == n_distinct
    assert paths[ZeroMean()][:2] == (
        "models.gauss.mean",
        "models.fhs.filter.mean",
    )
    assert paths[vol] == (
        "models.gauss.vol",
        "models.fhs.filter.vol",
        "backtests.gauss.kupiec.overshootings.var.vol",
        "backtests.fhs.kupiec.overshootings.var.filter.vol",
    )
    assert paths[gauss] == (
        "models.gauss",
        "backtests.gauss.kupiec.overshootings.var",
    )


def test_backtests_and_overshootings_have_paths_like_the_results() -> None:
    run, _, _ = _gauss_and_fhs()

    by_path = {path: n.node for n in run.graph().nodes for path in n.paths}

    assert isinstance(by_path["backtests.gauss.kupiec"], KupiecTest)
    assert isinstance(by_path["backtests.gauss.kupiec.overshootings"], Overshootings)
    assert by_path["backtests.fhs.kupiec"] is not by_path["backtests.gauss.kupiec"]


def test_the_nodes_come_from_the_models_then_the_backtests() -> None:
    run, _, gauss = _gauss_and_fhs()

    first_paths = [n.paths[0] for n in run.graph().nodes]

    assert first_paths[0] == "models.gauss"
    assert run.graph().nodes[0].node is gauss
    assert first_paths[-2:] == [
        "backtests.fhs.kupiec",
        "backtests.fhs.kupiec.overshootings",
    ]


def test_the_edges_go_from_each_dependency_to_its_reader() -> None:
    run, vol, gauss = _gauss_and_fhs()

    edges = set(run.graph().edges)

    assert GraphEdge(source=vol, target=gauss, name="vol") in edges
    # One edge per dependency of each distinct node.
    n_dependencies = sum(len(n.node.dependencies()) for n in run.graph().nodes)
    assert len(run.graph().edges) == len(edges) == n_dependencies
    nodes = {n.node for n in run.graph().nodes}
    assert all(e.source in nodes and e.target in nodes for e in edges)


def test_equal_models_built_separately_are_one_node() -> None:
    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        gauss = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="a")
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="b")

    assert _paths(ctx.graph())[gauss][:2] == ("models.a", "models.b")
    n_distinct = 5  # Model, volatility, mean, overshootings and backtest.
    assert len(ctx.graph().nodes) == n_distinct


def test_a_model_under_two_names_is_one_node_with_two_paths() -> None:
    gauss = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01))
    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        dataclasses.replace(gauss, name="a")
        dataclasses.replace(gauss, name="b")

    paths = _paths(ctx.graph())
    assert paths[gauss] == (
        "models.a",
        "models.b",
        "backtests.a.kupiec.overshootings.var",
        "backtests.b.kupiec.overshootings.var",
    )
    kupiec = next(n for n in ctx.graph().nodes if isinstance(n.node, KupiecTest))
    assert kupiec.paths == ("backtests.a.kupiec", "backtests.b.kupiec")


def test_a_context_without_backtests_has_only_its_models_graph() -> None:
    with Context() as ctx:
        mean = ZeroMean()
        vol = ConstantVolatility(0.01)
        gauss = GaussianModel(cf=0.99, vol=vol, mean=mean, name="gauss")

    assert set(_paths(ctx.graph())) == {gauss, vol, mean}


def test_results_hold_every_graph_node_without_recomputing() -> None:
    with Context(backtests={"k": KupiecTest}) as ctx:
        vol = _CountingVolatility(0.01)
        GaussianModel(cf=0.99, vol=vol, name="gauss")

    results = ctx.run(returns_series(n=100))
    graph_nodes = {entry.node for entry in ctx.graph().nodes}
    assert set(results.outputs) == graph_nodes
    for node in graph_nodes:
        assert results[node] is results.outputs[node]
    assert vol.n_calls == 1


def test_run_checks_graph_inputs_before_computing() -> None:
    with Context() as ctx:
        vol = _CountingVolatility(0.01)
        GaussianModel(cf=0.99, vol=vol, name="gauss")
        ValueAtRiskInput("var_ext", cf=0.99)
    returns = returns_series(n=100)

    with pytest.raises(KeyError, match=r"Missing inputs \['var_ext'\]"):
        ctx.run(returns, {"other": returns})
    assert vol.n_calls == 0
