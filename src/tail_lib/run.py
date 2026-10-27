"""Runs: models and their backtests, computed on one returns series.

A `Run` takes value-at-risk model nodes, and the backtests to run on each of
them (R3, R4, R5), then computes them on any returns series, with the run
inputs the models read (R13). Python code builds the nodes (R17);
`Run.from_spec` builds them from a `RunSpec`, e.g. one received as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

from pydantic import Field
from pydantic import model_validator

from tail_lib._checks import check_name
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.build import BacktestResult
from tail_lib.build import Builder
from tail_lib.graph import Cache
from tail_lib.graph import Pool
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import VolatilityInput
from tail_lib.specs import BacktestSpec
from tail_lib.specs import Label
from tail_lib.specs import Name
from tail_lib.specs import Spec
from tail_lib.specs import ValueAtRiskSpec
from tail_lib.specs import find_last_window
from tail_lib.windows import Last

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from tail_lib.graph import Node
    from tail_lib.models.value_at_risk import ValueAtRiskModel
    from tail_lib.time_series import TimeSeries
    from tail_lib.windows import Trailing


class RunSpec(Spec):
    """Spec of a run: named models, and named backtests run on each of them.

    The run inputs are not listed here: they are the input specs used in the
    models, e.g. `{"type": "input", "name": "var_ext", "cf": 0.99}`.

    A model with a `last` window, or with an input that has one, gives one
    value-at-risk, the forecast for a date after the data. It has nothing to
    backtest, so a run with backtests rejects it.

    `labels` gives display names to models and backtests, e.g. for a report
    (R20), as in `Run`.
    """

    models: Annotated[dict[Name, ValueAtRiskSpec], Field(min_length=1)]
    backtests: dict[Name, BacktestSpec] = Field(default_factory=dict)
    labels: dict[Name, Label] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_names_and_labels(self) -> RunSpec:
        _check_names_and_labels(self.models, self.backtests, self.labels)
        return self

    @model_validator(mode="after")
    def _check_backtested_models_have_a_history(self) -> RunSpec:
        if not self.backtests:
            return self
        for name, model in self.models.items():
            path = find_last_window(model)
            if path is not None:
                msg = (
                    f"Model {name!r} has a `last` window, at "
                    f"`models.{name}.{path}`, so it gives one forecast and can't "
                    f"be backtested, but the run has backtests "
                    f"{sorted(self.backtests)}!"
                )
                raise ValueError(msg)
        return self


@dataclass(frozen=True)
class RunResult:
    """The outputs of a run.

    Attributes:
        models (dict[str, TimeSeries]): The value-at-risk of each model, by
            model name.
        backtests (dict[str, dict[str, BacktestResult]]): The result of each
            backtest of each model, by model name, then backtest name.

    """

    models: dict[str, TimeSeries]
    backtests: dict[str, dict[str, BacktestResult]]


@dataclass(frozen=True)
class GraphNode:
    """A node of a run's graph, with every path it is reached by.

    Attributes:
        node (Node[Any]): The node.
        paths (tuple[str, ...]): Its paths from the run, e.g.
            `models.gauss.vol`, or `backtests.gauss.kupiec.overshootings`
            for a model's overshootings. A node shared by several readers has
            several (D3).

    """

    # The nodes of a graph have different output types: `Any` stands for each.
    node: Node[Any]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class GraphEdge:
    """An edge of a run's graph: a node reading another one.

    Attributes:
        source (Node[Any]): The node read, a dependency of `target`.
        target (Node[Any]): The node reading it.
        name (str): The name `target` reads it under, its key in
            `target.dependencies()`, e.g. `vol`.

    """

    source: Node[Any]
    target: Node[Any]
    name: str


@dataclass(frozen=True)
class RunGraph:
    """Every node of a run, each once, and the edges between them.

    Attributes:
        nodes (tuple[GraphNode, ...]): Each distinct node, in the order a
            depth-first walk reaches it: from each model, then from each
            backtest of each model.
        edges (tuple[GraphEdge, ...]): One edge per dependency of each node.

    """

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


type BacktestFunction = Callable[[Overshootings], Node[BacktestResult]]
"""Makes a backtest node from a model's overshootings, e.g. `KupiecTest` itself."""


class Run:
    """Models and their backtests, computed on any returns series.

    Models and backtests have distinct names, which can be given friendlier
    display names, `labels`, e.g. for a report (R20).

    Every backtest runs on every model (R4). Each model gets one
    `Overshootings` node, shared by all its backtests; a model node given
    under two names gets one too, and the same backtests, so they are
    computed once (D3).

    The run walks the graph of its models once, when built, through
    `Node.dependencies`, to find the inputs it reads (R13), and to reject
    backtests on a model with a `last` window, which gives one forecast, for
    a date after the data. Each input name belongs to one node: equal input
    nodes are one node (D9), but two different ones, e.g. a volatility input
    and a value-at-risk input, can't share a name. The whole graph, backtests
    included, is in `graph`.
    """

    def __init__(
        self,
        models: Mapping[str, ValueAtRiskModel],
        backtests: Mapping[str, BacktestFunction] | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Build a run from model nodes.

        Args:
            models (Mapping[str, ValueAtRiskModel]): The models, by name.
                Names, like backtest names, are Python identifiers.
            backtests (Mapping[str, BacktestFunction] | None): The backtests,
                by name, each as a function from a model's overshootings to
                the test node, e.g. `KupiecTest`, or
                `lambda ovs: KupiecTest(ovs, window=Trailing(length=250))`.
            labels (Mapping[str, str] | None): Display names, by model or
                backtest name, e.g. `{"gauss_99": "Gaussian 99%"}`. Names
                without one are shown as they are.

        Raises:
            ValueError: If there is no model, if a model or backtest name
                isn't a Python identifier, if a model and a backtest share a
                name, if a label is empty or names no model or backtest, if
                two different nodes read the
                same run input, or if there are backtests and a model has a
                `last` window, or reads a node that has one.

        """
        if not models:
            msg = "A run needs at least one model!"
            raise ValueError(msg)
        backtests = {} if backtests is None else backtests
        labels = {} if labels is None else labels
        # Names appear in paths through the graph, e.g. `models.gauss`.
        for name in models:
            check_name("Model", name)
        for name in backtests:
            check_name("Backtest", name)
        _check_names_and_labels(models, backtests, labels)
        self._labels = MappingProxyType(
            {name: labels.get(name, name) for name in [*models, *backtests]}
        )

        # Every node of each model, with its path from the run, e.g.
        # `models.gauss.vol.calibration`.
        graphs = {
            name: _graph(f"models.{name}", model) for name, model in models.items()
        }

        if backtests:
            _check_no_last_window(graphs, sorted(backtests))
        self._input_names = _input_names(graphs)

        self._models = Pool(dict(models))
        # One set of backtest nodes per distinct model node, on its
        # overshootings.
        tests: dict[ValueAtRiskModel, dict[str, Node[BacktestResult]]] = {}
        for model in models.values():
            if model not in tests:
                ovs = Overshootings(model)
                tests[model] = {name: make(ovs) for name, make in backtests.items()}
        self._backtests = {name: Pool(tests[model]) for name, model in models.items()}

        # The graph is walked from each model, then from each of its
        # backtests, which reach the model again through its overshootings.
        # A graph's nodes have different output types: `Any` stands for each.
        roots: list[tuple[str, Node[Any]]] = [
            (f"models.{name}", model) for name, model in models.items()
        ]
        roots += [
            (f"backtests.{model_name}.{test_name}", test)
            for model_name, model in models.items()
            for test_name, test in tests[model].items()
        ]
        self._graph = _run_graph(roots)

    @classmethod
    def from_spec(cls, spec: RunSpec) -> Run:
        """Build a run from a spec, e.g. one received as JSON (R9, R17).

        A `Builder` turns the specs into nodes: identical parts of the specs
        give one node (D3), e.g. a calibration shared by two models.

        Args:
            spec (RunSpec): The run spec.

        Returns:
            Run: The run of the spec's nodes.

        Raises:
            ValueError: If a value is out of a node's range, or an input is
                declared twice, differently.

        """
        builder = Builder()
        models = {name: builder.var_model(model) for name, model in spec.models.items()}
        backtests: dict[str, BacktestFunction] = {
            name: partial(builder.backtest, backtest)
            for name, backtest in spec.backtests.items()
        }
        return cls(models, backtests, spec.labels)

    @property
    def labels(self) -> Mapping[str, str]:
        """The display name of every model and backtest, by name.

        A name without a label is its own display name.
        """
        return self._labels

    @property
    def input_names(self) -> frozenset[str]:
        """The names of the inputs the run reads, to pass to `run`."""
        return self._input_names

    @property
    def graph(self) -> RunGraph:
        """Every node of the run, each once, with its paths and edges."""
        return self._graph

    def run(
        self,
        returns: TimeSeries,
        inputs: Mapping[str, TimeSeries] | None = None,
    ) -> RunResult:
        """Compute the models, then their backtests, on one returns series.

        Args:
            returns (TimeSeries): The returns time series.
            inputs (Mapping[str, TimeSeries] | None): The run inputs, by name,
                each with the dates and shape of `returns`. Inputs the run
                does not read are ignored.

        Returns:
            RunResult: The value-at-risk of each model, and its backtests.

        Raises:
            KeyError: If an input the run reads is missing. This is checked
                before computing anything.

        """
        return self.evaluate(Cache(returns, inputs))

    def evaluate(self, cache: Cache) -> RunResult:
        """Compute the models, then their backtests, on a cache.

        Every node of the run is then in the cache, so reading a node of
        `graph` from it costs nothing.

        Args:
            cache (Cache): The cache, holding the returns and the run inputs.
                It can be shared, e.g. with pools.

        Returns:
            RunResult: The value-at-risk of each model, and its backtests.

        Raises:
            KeyError: If an input the run reads is missing from the cache.
                This is checked before computing anything.

        """
        missing = self._input_names - cache.input_names
        if missing:
            msg = (
                f"Missing inputs {sorted(missing)}! Inputs given: "
                f"{sorted(cache.input_names)}."
            )
            raise KeyError(msg)

        models = self._models.evaluate(cache)
        backtests = {
            model_name: pool.evaluate(cache)
            for model_name, pool in self._backtests.items()
        }
        return RunResult(models=models, backtests=backtests)


def _check_names_and_labels(
    models: Mapping[str, object],
    backtests: Mapping[str, object],
    labels: Mapping[str, str],
) -> None:
    """Check models and backtests have distinct names, and every label a name.

    Args:
        models (Mapping[str, object]): The models, or their specs, by name.
        backtests (Mapping[str, object]): The backtests, or their specs, by
            name.
        labels (Mapping[str, str]): The display names, by model or backtest
            name.

    Raises:
        ValueError: If a model and a backtest share a name, or if a label is
            empty or names no model or backtest.

    """
    shared = models.keys() & backtests.keys()
    if shared:
        msg = f"Names {sorted(shared)} are both a model and a backtest!"
        raise ValueError(msg)

    unknown = labels.keys() - models.keys() - backtests.keys()
    if unknown:
        msg = (
            f"Labels {sorted(unknown)} name no model or backtest! Models: "
            f"{sorted(models)}, backtests: {sorted(backtests)}."
        )
        raise ValueError(msg)

    empty = [name for name, label in labels.items() if not label]
    if empty:
        msg = f"Labels of {sorted(empty)} are empty!"
        raise ValueError(msg)


type _Graph = list[tuple[str, Node[Any]]]
"""Every node of a model, with its path from the run, see `_graph`."""


def _check_no_last_window(graphs: Mapping[str, _Graph], backtests: list[str]) -> None:
    """Check no model has a `last` window, as a run with backtests needs.

    Args:
        graphs (Mapping[str, _Graph]): The graph of each model, by name.
        backtests (list[str]): The names of the run's backtests, for the
            message.

    Raises:
        ValueError: If a model, or a node it reads, has a `last` window.

    """
    for name, graph in graphs.items():
        last = [path for path, node in graph if isinstance(_window(node), Last)]
        if last:
            msg = (
                f"Model {name!r} has a `last` window, at `{last[0]}.window`, so "
                f"it gives one forecast and can't be backtested, but the run "
                f"has backtests {backtests}!"
            )
            raise ValueError(msg)


def _input_names(graphs: Mapping[str, _Graph]) -> frozenset[str]:
    """Find the run inputs the models read, each name belonging to one node.

    Equal input nodes are one node (D9). Two different nodes can't read one
    name, so no node silently reads another one's input, e.g. volatilities
    read as a value-at-risk.

    Args:
        graphs (Mapping[str, _Graph]): The graph of each model, by name.

    Returns:
        frozenset[str]: The names of the run inputs.

    Raises:
        ValueError: If two different nodes read the same run input.

    """
    input_nodes: dict[str, tuple[str, Node[Any]]] = {}
    for graph in graphs.values():
        for path, node in graph:
            input_name = _input_name(node)
            if input_name is None:
                continue
            first_path, first_node = input_nodes.setdefault(input_name, (path, node))
            if first_node != node:
                msg = (
                    f"Run input {input_name!r} is read by two different nodes, "
                    f"at `{first_path}` and `{path}`: give each input its own "
                    f"name!"
                )
                raise ValueError(msg)
    return frozenset(input_nodes)


# A node's dependencies have different output types: `Any` stands for each.
def _graph(path: str, node: Node[Any]) -> _Graph:
    """List a node and every node it reads, depth first, with their paths.

    Args:
        path (str): The path of `node`, e.g. `models.gauss`.
        node (Node[Any]): The node.

    Returns:
        _Graph: Each node, with its path: the path of the
            node that reads it, then the name it is read under, e.g.
            `models.gauss.vol`. A node read twice is listed twice.

    """
    graph = [(path, node)]
    for name, dependency in node.dependencies().items():
        graph += _graph(f"{path}.{name}", dependency)
    return graph


def _run_graph(roots: list[tuple[str, Node[Any]]]) -> RunGraph:
    """Gather the graph of a run from its roots, each node once.

    Args:
        roots (list[tuple[str, Node[Any]]]): The models and backtests, each
            with its path, e.g. `models.gauss` or `backtests.gauss.kupiec`.

    Returns:
        RunGraph: Each distinct node with its paths, in depth-first order
            from the roots, and the edges to each node from its
            dependencies.

    """
    # Nodes are identified by value, as in the cache (D9), and dicts keep the
    # order they are first reached in.
    paths: dict[Node[Any], list[str]] = {}
    for root_path, root in roots:
        for path, node in _graph(root_path, root):
            paths.setdefault(node, []).append(path)

    nodes = tuple(
        GraphNode(node, tuple(node_paths)) for node, node_paths in paths.items()
    )
    edges = tuple(
        GraphEdge(source=dependency, target=node, name=name)
        for node in paths
        for name, dependency in node.dependencies().items()
    )
    return RunGraph(nodes=nodes, edges=edges)


def _input_name(node: Node[Any]) -> str | None:
    """Give the name of the run input a node reads, if it is an input node."""
    match node:
        case ValueAtRiskInput() | VolatilityInput():
            return node.name
        case _:
            return None


def _window(node: Node[Any]) -> Trailing | Last | None:
    """Give the window of a node computed on windows, if it is one.

    Backtests are not listed: their windows are on the overshootings, and a
    `last` one is fine, e.g. Kupiec on the last 250 dates.
    """
    match node:
        case (
            HistoricalSimulationsModel()
            | FilteredHistoricalSimulationsModel()
            | EwmaDecayCalibration()
        ):
            return node.window
        case _:
            return None
