"""Model-building contexts.

`tail-lib` is built around the idea of running multiple models and backtests in
a single run. To do so, models and their components are built into a graph,
which is wired automatically as long as these components are written in a
given `Context`.

Let us consider the following code:

```python
with Context() as ctx:
    df = ConstantDecay(0.94)
    vol = EwmaVolatility(df)
    gauss = GaussianModel(cf=0.99, vol=vol, name="gauss")
    hs = HistoricalSimulationsModel(cf=0.99, window=Trailing(250), name="hs")
    kupiec_250 = Kupiec(window=Trailing(250))
```

This context would turn into a graph like the one below. (The overshooting
nodes are constructed automatically to feed into the )

```mermaid
flowchart TD
    df["ConstantDecay(0.94)"]
    vol["EwmaVolatility(df=...)"]
    gauss["GaussianModel(cf=0.99, vol=..., name='gauss')"]
    hs["HistoricalSimulationsModel(cf=0.99, window=Trailing(250), name='hs')"]
    os1{{"Overshootings(var=...)"}}
    os2{{"Overshootings(var=...)"}}
    kupiec_250["Kupiec(window=Trailing(250))"]

    df --> vol --> gauss --> os1 --> kupiec_250
    hs --> os2 --> kupiec_250
```

When running the context (via `ctx.run`), all these nodes get evaluated, and
the VaR model results will be available, along with the specified backtesting
results.

"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Self
from typing import overload
from uuid import uuid4

import numpy as np

from tail_lib._checks import check_name_is_identifier
from tail_lib._range import _check_model_history
from tail_lib._range import _range_slice
from tail_lib.backtests.base import BacktestDeclaration
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.build import _build_backtest
from tail_lib.build import _build_var_model
from tail_lib.build import _check_input_specs
from tail_lib.graph import Backtest
from tail_lib.graph import Node
from tail_lib.graph import RunInput
from tail_lib.graph import current_context
from tail_lib.models.ewma import DecayFactors
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import FilteredReturns
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import _first_full_date

if TYPE_CHECKING:
    import datetime
    from collections.abc import Iterable
    from collections.abc import Mapping
    from contextvars import Token
    from types import TracebackType

    from tail_lib.build import BacktestResult
    from tail_lib.run import RunSpec
    from tail_lib.time_series.windows import Trailing


@dataclass(frozen=True)
class GraphNode:
    """A node of a run's graph, with every path it is reached by.

    Attributes:
        node (Node[Any]): The node.
        paths (tuple[str, ...]): Its paths from the run, e.g.
            `models.gauss.vol`, or `backtests.gauss.kupiec.overshootings`
            for a model's overshootings. A node shared by several readers has
            several.

    """

    node: Node[Any]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class GraphEdge:
    """An edge of a run's graph.

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
            backtest of each model, then the other nodes bound to the context.
        edges (tuple[GraphEdge, ...]): One edge per dependency of each node.

    """

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


class Context:
    """A model-building context.

    In general, we want computation nodes to be bound to a given
    context. However, nodes can be declared outside of a context if they are
    meant to be re-used across multiple contexts. In such a case, when the
    context is run, the node is brought into the graph, even though it was not
    bound at the declaration stage.

    Nodes are identified by value, not by key. This allows users to define
    nodes in an anonymous way such as:

    ```python
    with Context() as ctx:
        vol = EwmaVolatility(decay=ConstantDecay(0.94))
        #                          ^^^^^^^^^^^^^^^^^^^
        #                          Graph node

        print(id(vol.decay)) # 4473043728

        vol_2 = EwmaVolatility(decay=ConstantDecay(0.94))
        print(id(vol.decay)) # 4473043728 - The same node.
    ```

    """

    def __init__(
        self,
        id_: str | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Open an empty context, to fill in a `with` block.

        Args:
            id_ (str | None): The id of the context. Defaults to a random
                UUID (v4).
            labels (Mapping[str, str] | None): Display names, by model or
                backtest name, e.g. `{"gauss": "Gaussian 99%"}`, e.g. for a
                report. Names without one are shown as they are.

        """
        self._id: str = id_ or str(uuid4())
        self._backtests: dict[str, BacktestDeclaration] = {}
        self._labels = {} if labels is None else dict(labels)
        # Every distinct node bound, by value, in the order first bound.
        self._nodes: dict[Node[Any], None] = {}
        # The named nodes bound, by name, and the models among them.
        self._named: dict[str, Node[Any]] = {}
        self._models: dict[str, ValueAtRiskModel] = {}
        # One token per open `with` block on this context, to restore the
        # context outside it on exit. A list, so blocks can nest.
        self._tokens: list[Token[Context | None]] = []

    def add_backtest(self, name: str, backtest: BacktestDeclaration) -> None:
        """Register a declaration to test every model of this context.

        Registration computes nothing and records no executable nodes.
        `graph()` binds the declaration to each model, including models
        added after the backtest. A declaration is immutable and can be
        reused in several contexts.

        Named declarations built inside a `with` block call this method
        automatically. For explicit registration, build the declaration
        outside the block. The `name` argument chooses its registered name.

        Args:
            name (str): The backtest's name, a Python identifier.
            backtest (BacktestDeclaration): The test and its settings, e.g.
                `Kupiec(window=Trailing(250))`.

        Raises:
            ValueError: If the name isn't an identifier, is already a
                registered backtest, or belongs to a named node.
            TypeError: If `backtest` isn't a declaration.

        """
        check_name_is_identifier("Backtest", name)
        if not isinstance(backtest, BacktestDeclaration):
            msg = f"Backtest {name!r} needs a backtest declaration, got {backtest!r}!"
            raise TypeError(msg)
        if name in self._backtests:
            msg = f"Backtest {name!r} is already registered!"
            raise ValueError(msg)
        if name in self._named:
            msg = f"Name {name!r} is both a node and a backtest!"
            raise ValueError(msg)
        self._backtests[name] = backtest

    @classmethod
    def from_spec(cls, spec: RunSpec) -> Self:
        """Build a context from a spec, e.g. received as JSON.

        The spec's model keys name the models. For a VaR input, the key
        has to match its input name, since the node has one name.
        Plain functions build each node and its inputs inside the context;
        equal nodes share computation by value. Every backtest in the
        spec runs on every model.

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

        with cls(labels=spec.labels) as ctx:
            for name, model in spec.models.items():
                _build_var_model(model, name=name)

            # Named declarations built in the block register on `ctx`.
            for name, backtest in spec.backtests.items():
                _build_backtest(backtest, name=name)

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
        *,
        lower: datetime.date | np.datetime64 | None = None,
        upper: datetime.date | np.datetime64 | None = None,
    ) -> Results:
        """Compute every node of the context once, on one returns series.

        The nodes are computed in an order where each node's inputs come
        before it. A run input's output is the input of its name, never
        computed. Every other output, but a backtest's, holds a time
        series with the dates and shape of the returns, checked after
        each node.

        With either bound, nodes still compute on the whole trimmed history.
        VaR is `NaN` outside the selected dates; preprocessors retain
        their history. Backtests receive only flags after the first selected
        forecast through the last selected date, inclusive. A single-date
        range has no observations to backtest.

        Args:
            returns (TimeSeries): The returns time series. Leading dates are
                removed until every portfolio has a value.
            inputs (Mapping[str, TimeSeries] | None): The run inputs, by name,
                each with the original dates and shape of `returns`. They
                are trimmed with the returns. Inputs no node reads are ignored.
            lower (datetime.date | np.datetime64 | None): Inclusive first
                forecast date. If a bound is given, omitted bounds use the
                first or last date of the trimmed history. All models must
                have finished their warm-up by the lower bound.
            upper (datetime.date | np.datetime64 | None): Inclusive last
                forecast date. Bounds select observations inside the history;
                they need not themselves be observations. With neither bound,
                the run keeps its full-history behavior, including warm-ups.

        Returns:
            Results: The output of every node, and the models' value-at-risk
                and backtests by name.

        Raises:
            KeyError: If a run input is missing, checked before computing.
            ValueError: As `graph`, or if a run input or a node's output
                does not match the returns, or no date has a return for
                every portfolio. Also if bounds are invalid, a model cannot
                cover the range, or a bounded VaR uses a step other than 1.
            TypeError: If a node other than a backtest outputs no time
                series.

        """
        graph = self.graph()
        inputs = {} if inputs is None else inputs
        input_nodes = [n.node for n in graph.nodes if isinstance(n.node, RunInput)]
        needed = {node.name for node in input_nodes}
        missing = needed - inputs.keys()
        if missing:
            msg = f"Missing inputs {sorted(missing)}! Inputs given: {sorted(inputs)}."
            raise KeyError(msg)

        # Check inputs on the original grid, before slicing can hide a
        # misalignment. Only inputs the graph reads are checked.
        inputs = {node.name: _run_input(node, returns, inputs) for node in input_nodes}
        first = _first_full_date(returns) if returns.n_dates else 0
        if first == returns.n_dates:
            msg = "Returns have no date with a value for every portfolio!"
            raise ValueError(msg)
        if first:
            returns = returns[first:]
            inputs = {name: series[first:] for name, series in inputs.items()}

        selection = None
        flags = None
        if lower is not None or upper is not None:
            selection = _range_slice(returns, lower, upper)
            _check_model_history(self._models, returns, inputs, selection.start)
            flags = _flag_range(selection, graph)

        # Nodes have different output types: `Results` restores the type of
        # each when read by node.
        outputs: dict[Node[Any], Any] = {}
        for node in _inputs_first(n.node for n in graph.nodes):
            if isinstance(node, RunInput):
                output = inputs[node.name]
            else:
                # The node's inputs come before it, so their outputs are known.
                node_inputs = {
                    name: outputs[dependency]
                    for name, dependency in node.dependencies().items()
                }
                node_returns = returns
                if isinstance(node, Backtest) and flags is not None:
                    node_returns = returns[flags]
                    node_inputs = _backtest_inputs(node_inputs, flags)
                output = node.compute(node_returns, **node_inputs)
                if not isinstance(node, Backtest):
                    _check_output(node, output, returns)
            if selection is not None:
                output = _select_output(node, output, selection)
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
            lower=None if selection is None else returns.dates[selection.start],  # pyright: ignore[reportArgumentType]
            upper=None if selection is None else returns.dates[selection.stop - 1],  # pyright: ignore[reportArgumentType]
        )

    def _backtest_nodes(self) -> dict[str, dict[str, Backtest[BacktestResult]]]:
        """Build each backtest node of each model, by model name, then test name.

        Equal models get equal overshootings and tests, so they are one node
        each. The nodes are built outside any context, so building them
        never binds them to this one, even inside its `with` block.
        """
        token = current_context.set(None)
        try:
            return {
                model_name: {
                    test_name: declaration._bind(model)  # noqa: SLF001
                    for test_name, declaration in self._backtests.items()
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
    they were computed on, so a report needs nothing else.

    Attributes:
        returns (TimeSeries): The returns the run computed on, starting at
            the first date with a value for every portfolio.
        models (dict[str, TimeSeries]): The value-at-risk of each model, by
            name. With explicit run bounds, values outside the requested
            range are `NaN`, on the full grid of `returns`.
        backtests (dict[str, dict[str, BacktestResult]]): The result of each
            backtest of each model, by model name, then backtest name.
        outputs (Mapping[Node[Any], Any]): The output of every node computed.
        named (Mapping[str, Node[Any]]): The nodes named in the context, by
            name: the nodes built in its block with a `name`.
        lower (np.datetime64 | None): The first selected forecast observation,
            after resolving the execution's inclusive bounds against the
            trimmed history. `None` for a run without bounds.
        upper (np.datetime64 | None): The last selected forecast observation.
            `None` for a run without bounds. These bounds belong to this
            execution; the context stores neither.

    """

    returns: TimeSeries
    models: dict[str, TimeSeries]
    backtests: dict[str, dict[str, BacktestResult]]
    # Nodes have different output types: `Any` stands for each.
    outputs: Mapping[Node[Any], Any]
    named: Mapping[str, Node[Any]]
    lower: np.datetime64 | None = None
    upper: np.datetime64 | None = None

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


def _flag_range(selection: slice, graph: RunGraph) -> slice:
    """Select flags starting one observation after the first forecast."""
    flags = slice(selection.start + 1, selection.stop)
    if flags.start == flags.stop and any(
        isinstance(entry.node, Backtest) for entry in graph.nodes
    ):
        msg = "A bounded run with backtests needs at least two forecast dates!"
        raise ValueError(msg)
    return flags


# A backtest's inputs have different output types, as in `Context.run`.
def _backtest_inputs(inputs: Mapping[str, Any], flags: slice) -> dict[str, Any]:
    """Give backtests only the selected flags; their outputs need no full grid."""
    return {
        name: value[flags] if isinstance(value, TimeSeries) else value
        for name, value in inputs.items()
    }


# Outputs differ by node, as in `Context.run` and `_check_output`.
def _select_output(node: Node[Any], output: Any, selection: slice) -> Any:  # noqa: ANN401
    """Mask forecasts and flags, keeping historical preprocessors intact."""
    if isinstance(node, ValueAtRiskModel):
        if not isinstance(output, TimeSeries):
            msg = f"{type(node).__name__} must output a TimeSeries of VaR!"
            raise TypeError(msg)
        if np.isnan(output.values[selection]).any():
            msg = f"Model {node.name!r} has missing VaR inside the requested range!"
            raise ValueError(msg)
        return _mask_dates(output, selection)
    if isinstance(node, Overshootings):
        return _mask_dates(output, slice(selection.start + 1, selection.stop))
    return output


def _mask_dates(series: TimeSeries, selection: slice) -> TimeSeries:
    """Keep the full grid, with NaN outside the selected observations."""
    values = series.values.copy()
    values[: selection.start] = np.nan
    values[selection.stop :] = np.nan
    return TimeSeries(dates=series.dates, values=values)


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
    """Check a node's output holds a time series aligned with the returns.

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
                f"time series: only backtests can!"
            )
            raise TypeError(msg)
    try:
        returns.check_aligned(series)
    except ValueError as error:
        msg = f"The output of {node_type} does not match the returns: {error}"
        raise ValueError(msg) from error


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

    Equal input nodes are one node. Two different nodes can't read one
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
    # Nodes are identified by value, and dicts keep the
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
    return node.name if isinstance(node, RunInput) else None


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
