"""Runs: models and their backtests, computed on one returns series.

A `Run` takes value-at-risk model nodes, and the backtests to run on each of
them (R3, R4, R5), then computes them on any returns series, with the run
inputs the models read (R13). Python code builds the nodes (R17);
`Run.from_spec` builds them from a `RunSpec`, e.g. one received as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

from pydantic import Field
from pydantic import model_validator

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
    """

    models: Annotated[dict[Name, ValueAtRiskSpec], Field(min_length=1)]
    backtests: dict[Name, BacktestSpec] = Field(default_factory=dict)

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


type BacktestFunction = Callable[[Overshootings], Node[BacktestResult]]
"""Makes a backtest node from a model's overshootings, e.g. `KupiecTest` itself."""


class Run:
    """Models and their backtests, computed on any returns series.

    Every backtest runs on every model (R4). Each model gets one
    `Overshootings` node, shared by all its backtests; a model node given
    under two names gets one too, and the same backtests, so they are
    computed once (D3).

    The run walks the graph of its models once, when built, through
    `Node.dependencies`, to find the inputs it reads (R13), and to reject
    backtests on a model with a `last` window, which gives one forecast, for
    a date after the data. Each input name belongs to one node: models that
    read the same input share that node.
    """

    def __init__(
        self,
        models: Mapping[str, ValueAtRiskModel],
        backtests: Mapping[str, BacktestFunction] | None = None,
    ) -> None:
        """Build a run from model nodes.

        Args:
            models (Mapping[str, ValueAtRiskModel]): The models, by name.
            backtests (Mapping[str, BacktestFunction] | None): The backtests,
                by name, each as a function from a model's overshootings to
                the test node, e.g. `KupiecTest`, or
                `lambda ovs: KupiecTest(ovs, window=Trailing(length=250))`.

        Raises:
            ValueError: If there is no model, if two different nodes read the
                same run input, or if there are backtests and a model has a
                `last` window, or reads a node that has one.

        """
        if not models:
            msg = "A run needs at least one model!"
            raise ValueError(msg)
        backtests = {} if backtests is None else backtests

        # Every node of each model, with its path from the run, e.g.
        # `models.gauss.vol_model.calibration`.
        graphs = {
            name: _graph(f"models.{name}", model) for name, model in models.items()
        }

        if backtests:
            _check_no_last_window(graphs, sorted(backtests))
        self._input_names = _input_names(graphs)

        self._models = Pool(dict(models))
        # One pool of backtests per distinct model node, on its overshootings.
        pools: dict[ValueAtRiskModel, Pool[BacktestResult]] = {}
        for model in models.values():
            if model not in pools:
                ovs = Overshootings(model)
                pools[model] = Pool(
                    {name: make(ovs) for name, make in backtests.items()}
                )
        self._backtests = {name: pools[model] for name, model in models.items()}

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
        return cls(models, backtests)

    @property
    def input_names(self) -> frozenset[str]:
        """The names of the inputs the run reads, to pass to `run`."""
        return self._input_names

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
        given = {} if inputs is None else inputs
        missing = self._input_names - given.keys()
        if missing:
            msg = f"Missing inputs {sorted(missing)}! Inputs given: {sorted(given)}."
            raise KeyError(msg)

        cache = Cache(returns, given)
        models = self._models.evaluate(cache)
        backtests = {
            model_name: pool.evaluate(cache)
            for model_name, pool in self._backtests.items()
        }
        return RunResult(models=models, backtests=backtests)


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

    Models that read the same input share its node, so no node silently reads
    another one's input.

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
            if first_node is not node:
                msg = (
                    f"Run input {input_name!r} is read by two different nodes, "
                    f"at `{first_path}` and `{path}`: share one node per input "
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
            `models.gauss.vol_model`. A node read twice is listed twice.

    """
    graph = [(path, node)]
    for name, dependency in node.dependencies().items():
        graph += _graph(f"{path}.{name}", dependency)
    return graph


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
