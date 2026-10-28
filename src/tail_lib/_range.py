"""Validation of a run's requested forecast range, before computing (R22)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from tail_lib.graph import Node
from tail_lib.graph import RunInput
from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import Trailing
from tail_lib.time_series.windows import _first_full_date

if TYPE_CHECKING:
    import datetime
    from collections.abc import Mapping

    from tail_lib.models.value_at_risk import ValueAtRiskModel
    from tail_lib.time_series.time_series import TimeSeries


def _range_slice(
    returns: TimeSeries,
    lower: datetime.date | np.datetime64 | None,
    upper: datetime.date | np.datetime64 | None,
) -> slice:
    """Resolve inclusive bounds to observations on the trimmed returns grid.

    Missing bounds use the first or last date. Bounds inside the history
    need not be observations, e.g. a weekend; the slice includes the dates
    between them. Bounds outside the history, or an empty range, fail.

    Raises:
        ValueError: If dates aren't unique and increasing, a bound is
            missing (`NaT`) or outside the history, or the range is empty.

    """
    dates = returns.dates
    if dates.dtype.kind != "M" or np.isnat(dates).any():
        msg = "A bounded run needs datetime64 dates without NaT!"
        raise ValueError(msg)
    if not np.all(dates[1:] > dates[:-1]):
        msg = "A bounded run needs unique, increasing dates!"
        raise ValueError(msg)

    if any(
        isinstance(bound, np.datetime64) and np.isnat(bound) for bound in (lower, upper)
    ):
        msg = "Run bounds cannot be NaT!"
        raise ValueError(msg)
    lo = dates[0] if lower is None else np.datetime64(lower)
    hi = dates[-1] if upper is None else np.datetime64(upper)
    if lo > hi:
        msg = f"Run lower bound {lo} is after upper bound {hi}!"
        raise ValueError(msg)
    if lo < dates[0] or hi > dates[-1]:
        msg = (
            f"Run bounds [{lo}, {hi}] are outside the trimmed history "
            f"[{dates[0]}, {dates[-1]}]!"
        )
        raise ValueError(msg)
    start = int(np.searchsorted(dates, lo, side="left"))
    stop = int(np.searchsorted(dates, hi, side="right"))
    if start == stop:
        msg = f"Run range [{lo}, {hi}] contains no observations!"
        raise ValueError(msg)
    return slice(start, stop)


def _check_model_history(
    models: Mapping[str, ValueAtRiskModel],
    returns: TimeSeries,
    inputs: Mapping[str, TimeSeries],
    start: int,
) -> None:
    """Check built-in models' warm-ups before any node computes.

    Only the first usable observation is propagated, inputs first. No node
    spans or calibration schedules depend on the requested range (R23).
    Unknown custom nodes inherit their inputs' first usable date; their
    actual forecast coverage is checked after computing.

    Raises:
        ValueError: If a model cannot forecast from `start`, a training
            window cannot initialize its EWMA, or a VaR uses a sparse step.

    """
    n = returns.n_dates
    # Nodes have different output types; the count applies to every type.
    firsts: dict[Node[Any], int] = {}

    def first(node: Node[Any], path: str) -> int:
        if node in firsts:
            return firsts[node]
        available = max(
            (first(dep, f"{path}.{name}") for name, dep in node.dependencies().items()),
            default=0,
        )
        if isinstance(node, RunInput):
            available = _first_full_date(inputs[node.name])
        available = _node_first_date(node, available, n, path)
        firsts[node] = available
        return available

    for name, model in models.items():
        available = first(model, f"models.{name}")
        if available > start:
            earliest = (
                f"first available forecast is {returns.dates[available]}"
                if available < n
                else f"needs at least {available + 1} returns, got {n}"
            )
            msg = (
                f"Model {name!r} cannot forecast from {returns.dates[start]}: "
                f"{earliest}, with history starting on {returns.dates[0]}!"
            )
            raise ValueError(msg)


# Outputs differ by node; only the first usable observation matters here.
def _node_first_date(node: Node[Any], available: int, n: int, path: str) -> int:
    """Add a built-in node's own warm-up to that of its inputs."""
    match node:
        case VolatilityFilter():
            return available + 1
        case EwmaVolatility(start=Backcast(window=window)):
            if n <= window:
                msg = f"`{path}` needs more than {window} returns for its backcast!"
                raise ValueError(msg)
            return max(available, window - 1)
        case HistoricalSimulationsModel() | FilteredHistoricalSimulationsModel():
            if isinstance(node.window, Trailing) and node.window.step != 1:
                msg = f"`{path}.window` needs step=1 for daily VaR in a bounded run!"
                raise ValueError(msg)
        case EwmaDecayCalibration(start=Backcast(window=window)):
            if node.window.length <= window:
                msg = (
                    f"`{path}.window` needs more than {window} dates for its backcast!"
                )
                raise ValueError(msg)
    if isinstance(
        node,
        (
            HistoricalSimulationsModel,
            FilteredHistoricalSimulationsModel,
            EwmaDecayCalibration,
        ),
    ):
        available += node.window.length - 1
        if isinstance(node.window, Last):
            available = max(available, n - 1)
    return available
