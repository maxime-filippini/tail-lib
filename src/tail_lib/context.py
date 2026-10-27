"""Contexts: nodes declared in Python or built from specs, run on any returns.

A `Context` will replace `Run` (step 5 of the context API migration, see
`docs/open-points.md`). Nodes built inside `with Context() as ctx:` bind to
it (D10). Its value-at-risk models are named, and every backtest given to the
context runs on every one of them (R4). `ctx.graph()` walks the graph without
computing; `ctx.run(returns, inputs)` computes every node once, and returns
`Results`.

`Context.from_spec(spec)` builds the same kind of context from a `RunSpec`,
with the spec's model keys as node names (R9, R17).

    with Context(backtests={"kupiec": KupiecTest}) as ctx:
        vol = EwmaVolatility(ConstantDecay(0.94))
        GaussianModel(cf=0.99, vol=vol, name="gauss")
        HistoricalSimulationsModel(cf=0.99, window=Trailing(250), name="hs")

    results = ctx.run(returns)
    results["gauss"]                     # the VaR of the Gaussian model
    results[vol]                         # the EWMA volatility
    results.backtests["gauss"]["kupiec"] # its Kupiec test
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Self
from typing import overload

from tail_lib._checks import check_name
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.build import _build_backtest
from tail_lib.build import _build_var_model
from tail_lib.build import _check_input_specs
from tail_lib.graph import Backtest
from tail_lib.graph import Node
from tail_lib.graph import RunInput
from tail_lib.graph import current_context
from tail_lib.models.ewma import DecayFactors
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import FilteredReturns
from tail_lib.run import _check_names_and_labels
from tail_lib.run import _check_no_last_window
from tail_lib.run import _graph
from tail_lib.run import _input_names
from tail_lib.run import _run_graph
from tail_lib.time_series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from contextvars import Token
    from types import TracebackType

    from tail_lib.build import BacktestResult
    from tail_lib.run import BacktestFunction
    from tail_lib.run import RunGraph
    from tail_lib.run import RunSpec


class Context:
    """The nodes declared in a `with Context():` block, and how to compute them.

    A node built inside the block binds to the context: the innermost open
    one of its thread or asyncio task (D10). A node built outside any context
    belongs to none, and can still be the input of a node in any context.
    Nodes are identified by value (D9), so a node built twice is one node,
    computed once.

    Every value-at-risk model built in the block needs a `name`, a Python
    identifier (R20): building one without a name raises `ValueError` there.
    Names are unique in a context: two different nodes can't share one, and
    no node can take a backtest's name. Equal models under two names are one
    node, given under both.

    Every backtest runs on every model (R4). Each model gets one
    `Overshootings` node, shared by all its backtests. A context with
    backtests rejects a model with a `last` window, which gives one forecast,
    for a date after the data.

    A context without models computes its other nodes, e.g. a volatility on
    its own.
    """

    def __init__(
        self,
        backtests: Mapping[str, BacktestFunction] | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Open an empty context, to fill in a `with` block.

        Args:
            backtests (Mapping[str, BacktestFunction] | None): The backtests,
                by name, each as a function from a model's overshootings to
                the test node, e.g. `KupiecTest`, or
                `lambda ovs: KupiecTest(ovs, window=Trailing(length=250))`.
            labels (Mapping[str, str] | None): Display names, by model or
                backtest name, e.g. `{"gauss": "Gaussian 99%"}`, e.g. for a
                report (R20). Names without one are shown as they are.

        Raises:
            ValueError: If a backtest name isn't a Python identifier.

        """
        self._backtests = {} if backtests is None else dict(backtests)
        for name in self._backtests:
            check_name("Backtest", name)
        self._labels = {} if labels is None else dict(labels)
        # Every distinct node bound, by value, in the order first bound.
        self._nodes: dict[Node[Any], None] = {}
        # The named nodes bound, by name, and the models among them.
        self._named: dict[str, Node[Any]] = {}
        self._models: dict[str, ValueAtRiskModel] = {}
        # One token per open `with` block on this context, to restore the
        # context outside it on exit. A list, so blocks can nest.
        self._tokens: list[Token[Context | None]] = []

    @classmethod
    def from_spec(cls, spec: RunSpec) -> Self:
        """Build a context from a spec, e.g. received as JSON (R9, R17).

        The spec's model keys name the models. For a VaR input, the key
        has to match its input name, since the node has one name (R13).
        Plain functions build each node and its inputs inside the context;
        equal nodes share computation by value (D9). Every backtest in the
        spec runs on every model (R4).

        Args:
            spec (RunSpec): The validated run spec.

        Returns:
            Self: A context ready to run on returns and named inputs.

        Raises:
            ValueError: If an input is declared twice, differently, a VaR
                input's model key differs from its input name, or the
                nodes violate the context's name or window rules.

        """
        _check_input_specs(spec.models.values())
        backtests: dict[str, BacktestFunction] = {
            name: partial(_build_backtest, backtest)
            for name, backtest in spec.backtests.items()
        }
        with cls(backtests=backtests, labels=spec.labels) as ctx:
            for name, model in spec.models.items():
                _build_var_model(model, name=name)
        ctx.graph()
        return ctx

    def __enter__(self) -> Self:
        """Make this context the one nodes bind to, until the block ends."""
        self._tokens.append(current_context.set(self))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Give back the context that was open before the block."""
        current_context.reset(self._tokens.pop())

    @staticmethod
    def current() -> Context | None:
        """Give the context of the innermost open `with` block, or `None`."""
        return current_context.get()

    def bind(self, node: Node[Any]) -> None:
        """Add a node to the context. A node built in the block calls it.

        Args:
            node (Node[Any]): The node, built in a `with` block on this
                context.

        Raises:
            ValueError: If the node is a value-at-risk model without a name,
                or if its name is a backtest's or another node's.

        """
        name = node.name
        if isinstance(node, ValueAtRiskModel) and name is None:
            msg = (
                f"{type(node).__name__} is built in a context, so it needs a "
                f'name, e.g. `name="gauss"`!'
            )
            raise ValueError(msg)
        if name is not None:
            if name in self._backtests:
                msg = f"Name {name!r} is both a node and a backtest!"
                raise ValueError(msg)
            # An equal node may take its name again, e.g. an input built twice.
            if self._named.setdefault(name, node) != node:
                msg = f"Name {name!r} is already used by another node!"
                raise ValueError(msg)
            if isinstance(node, ValueAtRiskModel):
                self._models[name] = node
        self._nodes.setdefault(node, None)

    @property
    def models(self) -> Mapping[str, ValueAtRiskModel]:
        """The value-at-risk models of the context, by name."""
        return MappingProxyType(self._models)

    @property
    def labels(self) -> Mapping[str, str]:
        """The display name of every model and backtest, by name.

        A name without a label is its own display name.
        """
        names = [*self._models, *self._backtests]
        return MappingProxyType({name: self._labels.get(name, name) for name in names})

    def graph(self) -> RunGraph:
        """Walk every node of the context, each once, without computing.

        The nodes are reached from each model (`models.gauss`), then from
        each of its backtests (`backtests.gauss.kupiec`), then from the
        other nodes bound to the context and not reached yet, by their name
        if they have one (`nodes.vol`), else by their class
        (`nodes.EwmaVolatility`, then `nodes.EwmaVolatility_2`, ...).

        Returns:
            RunGraph: Each distinct node with its paths, and the edges to
                each node from its dependencies.

        Raises:
            ValueError: If a model and a backtest share a name, if a label
                is empty or names no model or backtest, if there are
                backtests and a model has a `last` window, or reads a node
                that has one, or if two different nodes read the same run
                input.

        """
        _check_names_and_labels(self._models, self._backtests, self._labels)
        graphs = {
            name: _graph(f"models.{name}", model)
            for name, model in self._models.items()
        }
        if self._backtests:
            _check_no_last_window(graphs, sorted(self._backtests))

        roots: list[tuple[str, Node[Any]]] = [
            (f"models.{name}", model) for name, model in self._models.items()
        ]
        roots += [
            (f"backtests.{model_name}.{test_name}", test)
            for model_name, tests in self._backtest_nodes().items()
            for test_name, test in tests.items()
        ]
        roots += _other_roots(self._nodes, roots)

        # Every path's graph, to find the run inputs and check each input
        # name belongs to one node.
        _input_names({path: _graph(path, node) for path, node in roots})
        return _run_graph(roots)

    def run(
        self,
        returns: TimeSeries,
        inputs: Mapping[str, TimeSeries] | None = None,
    ) -> Results:
        """Compute every node of the context once, on one returns series.

        The nodes are computed in an order where each node's inputs come
        before it. A run input's output is the input of its name, never
        computed (R13). Every other output, but a backtest's, holds a time
        series with the dates and shape of the returns (R24), checked after
        each node.

        Args:
            returns (TimeSeries): The returns time series.
            inputs (Mapping[str, TimeSeries] | None): The run inputs, by name,
                each with the dates and shape of `returns`. Inputs no node
                reads are ignored.

        Returns:
            Results: The output of every node, and the models' value-at-risk
                and backtests by name.

        Raises:
            KeyError: If a run input is missing, checked before computing.
            ValueError: As `graph`, or if a run input or a node's output
                does not match the returns.
            TypeError: If a node other than a backtest outputs no time
                series.

        """
        graph = self.graph()
        inputs = {} if inputs is None else inputs
        needed = {n.node.name for n in graph.nodes if isinstance(n.node, RunInput)}
        missing = needed - inputs.keys()
        if missing:
            msg = f"Missing inputs {sorted(missing)}! Inputs given: {sorted(inputs)}."
            raise KeyError(msg)

        # Nodes have different output types: `Results` restores the type of
        # each when read by node.
        outputs: dict[Node[Any], Any] = {}
        for node in _inputs_first(n.node for n in graph.nodes):
            if isinstance(node, RunInput):
                output = _run_input(node, returns, inputs)
            else:
                # The node's inputs come before it, so their outputs are known.
                node_inputs = {
                    name: outputs[dependency]
                    for name, dependency in node.dependencies().items()
                }
                output = node.compute(returns, **node_inputs)
                if not isinstance(node, Backtest):
                    _check_output(node, output, returns)
            outputs[node] = output

        backtests = {
            model_name: {test_name: outputs[test] for test_name, test in tests.items()}
            for model_name, tests in self._backtest_nodes().items()
        }
        return Results(
            returns=returns,
            models={name: outputs[model] for name, model in self._models.items()},
            backtests=backtests,
            outputs=MappingProxyType(outputs),
            named=MappingProxyType(dict(self._named)),
        )

    def _backtest_nodes(self) -> dict[str, dict[str, Backtest[BacktestResult]]]:
        """Build each backtest node of each model, by model name, then test name.

        Equal models get equal overshootings and tests, so they are one node
        each (D9). The nodes are built outside any context, so building them
        never binds them to this one, even inside its `with` block.
        """
        token = current_context.set(None)
        try:
            return {
                model_name: {
                    test_name: make(Overshootings(model))
                    for test_name, make in self._backtests.items()
                }
                for model_name, model in self._models.items()
            }
        finally:
            current_context.reset(token)


@dataclass(frozen=True)
class Results:
    """The outputs of one run of a context.

    Read the output of any node by node, `results[vol]`, or of a node named
    in the context by its name, `results["gauss"]`. They hold the returns
    they were computed on, so a report needs nothing else (R19).

    Attributes:
        returns (TimeSeries): The returns the run computed on.
        models (dict[str, TimeSeries]): The value-at-risk of each model, by
            name.
        backtests (dict[str, dict[str, BacktestResult]]): The result of each
            backtest of each model, by model name, then backtest name.
        outputs (Mapping[Node[Any], Any]): The output of every node computed.
        named (Mapping[str, Node[Any]]): The nodes named in the context, by
            name: the nodes built in its block with a `name`.

    """

    returns: TimeSeries
    models: dict[str, TimeSeries]
    backtests: dict[str, dict[str, BacktestResult]]
    # Nodes have different output types: `Any` stands for each.
    outputs: Mapping[Node[Any], Any]
    named: Mapping[str, Node[Any]]

    @overload
    def __getitem__[T](self, key: Node[T]) -> T: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...  # noqa: ANN401

    def __getitem__(self, key: Node[Any] | str) -> Any:
        """Get the output of a node, or of the node of a name.

        Args:
            key (Node[Any] | str): The node, or its name in the context.

        Returns:
            Any: The output of the node, of its type when read by node.

        Raises:
            KeyError: If no node has this name, or the node wasn't computed.

        """
        if isinstance(key, str):
            if key not in self.named:
                msg = f"No node is named {key!r}! Names: {sorted(self.named)}."
                raise KeyError(msg)
            key = self.named[key]
        if key not in self.outputs:
            msg = f"{type(key).__name__} wasn't computed in this run!"
            raise KeyError(msg)
        return self.outputs[key]


def _other_roots(
    bound: Iterable[Node[Any]], roots: list[tuple[str, Node[Any]]]
) -> list[tuple[str, Node[Any]]]:
    """Give the bound nodes not reached from the roots, each with a path.

    The latest bound first: a node is built after its inputs, so starting
    from it reaches them, and they don't become roots of their own.

    Args:
        bound (Iterable[Node[Any]]): The nodes bound to the context, in the
            order bound.
        roots (list[tuple[str, Node[Any]]]): The roots so far, with their
            paths.

    Returns:
        list[tuple[str, Node[Any]]]: Each other root, with its path:
            `nodes.<name>`, or `nodes.<class>` for a node without a name,
            with `_2`, `_3`, ... after the first of a class.

    """
    reached = {node for path, root in roots for _, node in _graph(path, root)}
    others: list[tuple[str, Node[Any]]] = []
    seen_classes: dict[str, int] = {}
    for node in reversed(list(bound)):
        if node in reached:
            continue
        if node.name is not None:
            path = f"nodes.{node.name}"
        else:
            class_name = type(node).__name__
            count = seen_classes[class_name] = seen_classes.get(class_name, 0) + 1
            path = f"nodes.{class_name}" + ("" if count == 1 else f"_{count}")
        others.append((path, node))
        reached.update(node for _, node in _graph(path, node))
    return others


def _inputs_first(nodes: Iterable[Node[Any]]) -> list[Node[Any]]:
    """Order nodes so that each node's inputs come before it.

    Args:
        nodes (Iterable[Node[Any]]): The nodes, with every input of each.

    Returns:
        list[Node[Any]]: Each node once, after its dependencies.

    """
    order: dict[Node[Any], None] = {}

    def visit(node: Node[Any]) -> None:
        if node in order:
            return
        for dependency in node.dependencies().values():
            visit(dependency)
        order[node] = None

    for node in nodes:
        visit(node)
    return list(order)


def _run_input(
    node: RunInput, returns: TimeSeries, inputs: Mapping[str, TimeSeries]
) -> TimeSeries:
    """Give a run input node the input of its name, checked against the returns.

    Raises:
        ValueError: If the input does not match the returns.

    """
    value = inputs[node.name]
    try:
        returns.check_aligned(value)
    except ValueError as error:
        msg = f"Input {node.name!r} does not match the returns: {error}"
        raise ValueError(msg) from error
    return value


# Outputs have different types: `Any` stands for each.
def _check_output(node: Node[Any], output: Any, returns: TimeSeries) -> None:  # noqa: ANN401
    """Check a node's output holds a time series aligned with the returns (R24).

    A `DecayFactors` holds its `values`, and `FilteredReturns` its
    `standardized` returns, which it checks its mean and volatility against.

    Raises:
        TypeError: If the output holds no time series.
        ValueError: If its time series does not match the returns.

    """
    node_type = type(node).__name__
    match output:
        case TimeSeries():
            series = output
        case DecayFactors(values=series) | FilteredReturns(standardized=series):
            pass
        case _:
            msg = (
                f"{node_type} outputs a {type(output).__name__}, which holds no "
                f"time series: only backtests can (R24)!"
            )
            raise TypeError(msg)
    try:
        returns.check_aligned(series)
    except ValueError as error:
        msg = f"The output of {node_type} does not match the returns: {error}"
        raise ValueError(msg) from error
