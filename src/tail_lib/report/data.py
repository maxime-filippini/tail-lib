"""The data of a run report: one JSON document, the contract with the page (R19).

`ReportData` holds everything the report page shows: the returns, the run's
graph, the output of every time-series node, and the backtest results. Its
JSON Schema is also the source of the page's TypeScript types, so both sides
are checked against one contract (R11). The data service could serve it as
it is (R17).

Conventions in the data:

- A node is identified by its first path in the run's graph, e.g.
  `models.gauss.vol` (see `Run.graph`).
- Series are flattened: the batch axes of the returns become one list of
  series, in C order, each with a label. Every series of values is a list
  over the series, then over the dates (or the windows of a backtest).
- `NaN` becomes `null`, and dates are ISO strings, e.g. `2024-01-31`.
- Settings that are not numbers or text are written as in the specs (R9),
  e.g. `{"type": "trailing", "length": 250, "step": 1}`.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import assert_never

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import JsonValue

import tail_lib
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.graph import Preprocessor
from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import DecayFactors
from tail_lib.models.ewma import DecayModel
from tail_lib.models.ewma import StartingVol
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.run import _input_name
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Last
from tail_lib.windows import Trailing

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tail_lib._typing import DateArray1D
    from tail_lib.backtests.kupiec import KupiecResult
    from tail_lib.graph import Cache
    from tail_lib.graph import Node
    from tail_lib.graph import Setting
    from tail_lib.run import Run

type NodeKind = Literal["preprocessor", "model", "overshootings", "backtest"]
"""The role of a node in a run, which sets its layer in the graph."""

type Values = list[list[float | None]]
"""Values by series, then by date (or window). `NaN` is `None`."""


class _Data(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Metadata(_Data):
    """Where the report comes from.

    Attributes:
        title (str): The title of the report.
        created (datetime.datetime): When the report data was built.
        version (str): The version of tail-lib that built it.

    """

    title: str
    created: datetime.datetime
    version: str


class NodeData(_Data):
    """A node of the run's graph.

    Attributes:
        id (str): The node's first path in the run's graph.
        kind (NodeKind): The node's role in the run.
        class_name (str): The node's class, e.g. `EwmaVolatility`.
        settings (dict[str, JsonValue]): The node's settings, see
            `Node.settings`, written as JSON.
        paths (list[str]): Every path the node is reached by.
        input_name (str | None): The run input the node reads, if any.

    """

    id: str
    kind: NodeKind
    class_name: str
    settings: dict[str, JsonValue]
    paths: list[str]
    input_name: str | None


class EdgeData(_Data):
    """An edge of the run's graph: `target` reads `source` as `name`.

    Attributes:
        source (str): The id of the node read.
        target (str): The id of the node reading it.
        name (str): The name it is read under, e.g. `vol`.

    """

    source: str
    target: str
    name: str


class KupiecData(_Data):
    """A Kupiec test result, see `KupiecResult`, one value per window.

    Attributes:
        type (Literal["kupiec"]): The kind of backtest.
        end_dates (list[str]): The last date of each window.
        expected_rate (float): The overshooting rate under the model.
        n_obs (Values): The number of dates with a flag.
        n_overshootings (Values): The number of overshootings.
        observed_rate (Values): The observed overshooting rate.
        statistic (Values): The likelihood-ratio statistic.
        p_value (Values): The p-value of the test.

    """

    type: Literal["kupiec"] = "kupiec"
    end_dates: list[str]
    expected_rate: float
    n_obs: Values
    n_overshootings: Values
    observed_rate: Values
    statistic: Values
    p_value: Values


class BacktestData(_Data):
    """A backtest of a model: its node and its result.

    Attributes:
        node (str): The id of the backtest node.
        result (KupiecData): The result.

    """

    node: str
    result: KupiecData


class ReportData(_Data):
    """Everything a run report shows, as one JSON document.

    Attributes:
        metadata (Metadata): The title, creation time and version.
        dates (list[str]): The dates of the returns.
        series (list[str]): The label of each series.
        returns (Values): The returns.
        labels (dict[str, str]): The display name of each model and
            backtest, see `Run.labels`.
        models (dict[str, str]): The node id of each model, by model name.
        nodes (list[NodeData]): Each node of the run's graph, once.
        edges (list[EdgeData]): The edges of the run's graph.
        outputs (dict[str, Values]): The output of each time-series node,
            by node id: preprocessors, models and overshootings.
        backtests (dict[str, dict[str, BacktestData]]): The backtests, by
            model name, then backtest name.

    """

    metadata: Metadata
    dates: list[str]
    series: list[str]
    returns: Values
    labels: dict[str, str]
    models: dict[str, str]
    nodes: list[NodeData]
    edges: list[EdgeData]
    outputs: dict[str, Values]
    backtests: dict[str, dict[str, BacktestData]]


def build_report_data(
    run: Run,
    cache: Cache,
    *,
    title: str = "tail-lib report",
    series_labels: Sequence[str] | None = None,
) -> ReportData:
    """Compute a run on a cache, and gather the data of its report.

    The run is computed with `run.evaluate(cache)`, so every node of its
    graph is in the cache, and is read from it without computing it again.

    Args:
        run (Run): The run.
        cache (Cache): The cache, holding the returns and the run inputs.
        title (str): The title of the report.
        series_labels (Sequence[str] | None): A label per series of the
            returns, in C order over the batch axes. By default, each series
            is labelled by its index, e.g. `0`, or `(0, 1)` with several
            batch axes.

    Returns:
        ReportData: The data of the report.

    Raises:
        KeyError: If an input the run reads is missing from the cache.
        ValueError: If there are not as many series labels as series.

    """
    result = run.evaluate(cache)
    returns = cache.returns
    batch_shape = returns.values.shape[1:]

    ids = {graph_node.node: graph_node.paths[0] for graph_node in run.graph.nodes}
    nodes = [
        NodeData(
            id=ids[graph_node.node],
            kind=_kind(graph_node.node),
            class_name=type(graph_node.node).__name__,
            settings={
                name: _setting_json(value)
                for name, value in graph_node.node.settings().items()
            },
            paths=list(graph_node.paths),
            input_name=_input_name(graph_node.node),
        )
        for graph_node in run.graph.nodes
    ]
    edges = [
        EdgeData(source=ids[edge.source], target=ids[edge.target], name=edge.name)
        for edge in run.graph.edges
    ]

    # Every node is already in the cache: `get` only reads it. A decay model
    # gives its decay factors with their refits: the page charts the factors.
    # Backtests don't give a time series: their results are in `backtests`.
    outputs: dict[str, Values] = {}
    for graph_node in run.graph.nodes:
        output = cache.get(graph_node.node)
        if isinstance(output, DecayFactors):
            output = output.values
        if isinstance(output, TimeSeries):
            outputs[ids[graph_node.node]] = _values(output.values)

    # Models and backtests are found by their paths, e.g. `models.gauss` and
    # `backtests.gauss.kupiec`.
    by_path = {
        path: graph_node.node
        for graph_node in run.graph.nodes
        for path in graph_node.paths
    }
    backtests = {
        model_name: {
            test_name: BacktestData(
                node=ids[by_path[f"backtests.{model_name}.{test_name}"]],
                result=_kupiec_data(test_result),
            )
            for test_name, test_result in model_tests.items()
        }
        for model_name, model_tests in result.backtests.items()
    }

    return ReportData(
        metadata=Metadata(
            title=title,
            created=datetime.datetime.now(tz=datetime.UTC),
            version=tail_lib.__version__,
        ),
        dates=_dates(returns.dates),
        series=_series_labels(batch_shape, series_labels),
        returns=_values(returns.values),
        labels=dict(run.labels),
        models={name: ids[by_path[f"models.{name}"]] for name in result.models},
        nodes=nodes,
        edges=edges,
        outputs=outputs,
        backtests=backtests,
    )


# The nodes of a graph have different output types: `Any` stands for each.
def _kind(node: Node[Any]) -> NodeKind:
    """Give the role of a node in a run."""
    match node:
        case Overshootings():
            return "overshootings"
        case ValueAtRiskModel():
            return "model"
        case Preprocessor() | DecayModel():
            return "preprocessor"
        case _:
            # A run's other nodes are its backtests.
            return "backtest"


def _setting_json(value: Setting) -> JsonValue:
    """Write a setting as JSON, in the form the specs use."""
    match value:
        case Trailing(length=length, step=step):
            return {"type": "trailing", "length": length, "step": step}
        case Last(length=length):
            return {"type": "last", "length": length}
        case Backcast(window=window):
            return {"type": "backcast", "window": window}
        case StartingVol(vol=vol):
            return {"type": "starting_vol", "vol": _array_json(vol)}
        case tuple():
            # Bounds, or an array of values per series as nested tuples.
            return _array_json(value)
        case None | str() | float() | int():
            return value
        case _:
            assert_never(value)


def _array_json(value: Any) -> JsonValue:  # noqa: ANN401 - any array-like.
    """Write an array as a number, or a nested list for one value per series."""
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array.tolist()


def _values(values: np.ndarray) -> Values:
    """Flatten values of shape (n, *batch) to a list per series, NaN as None."""
    by_series = values.reshape(values.shape[0], -1).T.astype(float)
    return [
        [None if np.isnan(v) else v for v in series.tolist()] for series in by_series
    ]


def _dates(dates: DateArray1D) -> list[str]:
    """Write dates as ISO strings, e.g. `2024-01-31`."""
    return np.datetime_as_string(dates, unit="D").tolist()


def _series_labels(
    batch_shape: tuple[int, ...], labels: Sequence[str] | None
) -> list[str]:
    """Label each series, by the given labels or by its index."""
    n_series = int(np.prod(batch_shape))
    if labels is not None:
        if len(labels) != n_series:
            msg = f"Got {len(labels)} series labels for {n_series} series!"
            raise ValueError(msg)
        return list(labels)
    match len(batch_shape):
        case 0:
            return ["0"]
        case 1:
            return [str(i) for i in range(n_series)]
        case _:
            return [str(index) for index in np.ndindex(batch_shape)]


def _kupiec_data(result: KupiecResult) -> KupiecData:
    """Write a Kupiec result as report data."""
    return KupiecData(
        end_dates=_dates(result.end_dates),
        expected_rate=result.expected_rate,
        n_obs=_values(result.n_obs),
        n_overshootings=_values(result.n_overshootings),
        observed_rate=_values(result.observed_rate),
        statistic=_values(result.statistic),
        p_value=_values(result.p_value),
    )
