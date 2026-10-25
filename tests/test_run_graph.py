from typing import Any

import numpy as np
import pytest

from tail_lib.backtests.kupiec import KupiecTest
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.graph import Cache
from tail_lib.graph import Node
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.run import GraphEdge
from tail_lib.run import Run
from tail_lib.run import RunGraph
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


def _gauss_and_fhs() -> tuple[Run, ConstantVolatility, GaussianModel]:
    vol = ConstantVolatility(0.01)
    gauss = GaussianModel(cf=0.99, vol_model=vol)
    fhs = FilteredHistoricalSimulationsModel(
        cf=0.99, filter=VolatilityFilter(vol), window=Trailing(length=50)
    )
    run = Run({"gauss": gauss, "fhs": fhs}, {"kupiec": KupiecTest})
    return run, vol, gauss


def test_the_graph_lists_each_node_once_with_all_its_paths() -> None:
    run, vol, gauss = _gauss_and_fhs()

    paths = _paths(run.graph)

    # Each distinct node once: 2 models, the filter, the shared volatility,
    # 2 zero means, then per model one overshootings node and one backtest.
    n_distinct = 2 + 1 + 1 + 2 + 2 * 2
    assert len(run.graph.nodes) == len(paths) == n_distinct
    assert paths[vol] == (
        "models.gauss.vol_model",
        "models.fhs.filter.vol_model",
        "models.fhs.vol_model",
        "backtests.gauss.kupiec.overshootings.var_model.vol_model",
        "backtests.fhs.kupiec.overshootings.var_model.filter.vol_model",
        "backtests.fhs.kupiec.overshootings.var_model.vol_model",
    )
    assert paths[gauss] == (
        "models.gauss",
        "backtests.gauss.kupiec.overshootings.var_model",
    )


def test_backtests_and_overshootings_have_paths_like_the_results() -> None:
    run, _, _ = _gauss_and_fhs()

    by_path = {path: n.node for n in run.graph.nodes for path in n.paths}

    assert isinstance(by_path["backtests.gauss.kupiec"], KupiecTest)
    assert isinstance(by_path["backtests.gauss.kupiec.overshootings"], Overshootings)
    assert by_path["backtests.fhs.kupiec"] is not by_path["backtests.gauss.kupiec"]


def test_the_nodes_come_from_the_models_then_the_backtests() -> None:
    run, _, gauss = _gauss_and_fhs()

    first_paths = [n.paths[0] for n in run.graph.nodes]

    assert first_paths[0] == "models.gauss"
    assert run.graph.nodes[0].node is gauss
    assert first_paths[-2:] == [
        "backtests.fhs.kupiec",
        "backtests.fhs.kupiec.overshootings",
    ]


def test_the_edges_go_from_each_dependency_to_its_reader() -> None:
    run, vol, gauss = _gauss_and_fhs()

    edges = set(run.graph.edges)

    assert GraphEdge(source=vol, target=gauss, name="vol_model") in edges
    # One edge per dependency of each distinct node.
    n_dependencies = sum(len(n.node.dependencies()) for n in run.graph.nodes)
    assert len(run.graph.edges) == len(edges) == n_dependencies
    nodes = {n.node for n in run.graph.nodes}
    assert all(e.source in nodes and e.target in nodes for e in edges)


def test_a_model_under_two_names_is_one_node_with_two_paths() -> None:
    gauss = GaussianModel(cf=0.99, vol_model=ConstantVolatility(0.01))
    run = Run({"a": gauss, "b": gauss}, {"kupiec": KupiecTest})

    paths = _paths(run.graph)

    assert paths[gauss] == (
        "models.a",
        "models.b",
        "backtests.a.kupiec.overshootings.var_model",
        "backtests.b.kupiec.overshootings.var_model",
    )
    kupiec = next(n for n in run.graph.nodes if isinstance(n.node, KupiecTest))
    assert kupiec.paths == ("backtests.a.kupiec", "backtests.b.kupiec")


def test_a_run_without_backtests_has_only_its_models_graph() -> None:
    mean = ZeroMean()
    vol = ConstantVolatility(0.01)
    run = Run({"gauss": GaussianModel(cf=0.99, vol_model=vol, mean_model=mean)})

    assert set(_paths(run.graph)) == {run.graph.nodes[0].node, vol, mean}


def test_evaluate_leaves_every_graph_node_in_the_cache() -> None:
    vol = _CountingVolatility(0.01)
    run = Run({"gauss": GaussianModel(cf=0.99, vol_model=vol)}, {"k": KupiecTest})
    cache = Cache(returns_series(n=100))

    result = run.evaluate(cache)

    for graph_node in run.graph.nodes:
        cache.get(graph_node.node)
    assert vol.n_calls == 1
    np.testing.assert_array_equal(
        result.models["gauss"].values,
        run.run(returns_series(n=100)).models["gauss"].values,
    )


def test_evaluate_checks_the_inputs_of_the_cache_before_computing() -> None:
    vol = _CountingVolatility(0.01)
    run = Run(
        {
            "gauss": GaussianModel(cf=0.99, vol_model=vol),
            "ext": ValueAtRiskInput("var_ext", cf=0.99),
        }
    )
    returns = returns_series(n=100)

    with pytest.raises(KeyError, match=r"Missing inputs \['var_ext'\]"):
        run.evaluate(Cache(returns, {"other": returns}))
    assert vol.n_calls == 0


def test_a_cache_lists_its_input_names() -> None:
    returns = returns_series(n=10)

    assert Cache(returns).input_names == frozenset()
    assert Cache(returns, {"a": returns, "b": returns}).input_names == {"a", "b"}
