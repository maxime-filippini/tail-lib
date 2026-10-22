"""Runs: models and their backtests, from one spec, on one returns series.

A `RunSpec` names the value-at-risk models to compute and the backtests to
run on each of them (R3, R4, R5). `Run` builds its nodes once, then computes
them on any returns series, with the run inputs the specs read (R13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Annotated

from pydantic import Field

from tail_lib.build import BacktestResult
from tail_lib.build import Builder
from tail_lib.graph import Cache
from tail_lib.graph import Pool
from tail_lib.specs import BacktestSpec
from tail_lib.specs import Name
from tail_lib.specs import Spec
from tail_lib.specs import ValueAtRiskSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tail_lib.time_series import TimeSeries


class RunSpec(Spec):
    """Spec of a run: named models, and named backtests run on each of them.

    The run inputs are not listed here: they are the input specs used in the
    models, e.g. `{"type": "input", "name": "var_ext", "cf": 0.99}`.
    """

    models: Annotated[dict[Name, ValueAtRiskSpec], Field(min_length=1)]
    backtests: dict[Name, BacktestSpec] = Field(default_factory=dict)


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


class Run:
    """The nodes of a run spec, built once and computed on any returns series.

    Identical parts of the specs give one node (see `Builder`), so a run
    computes them once, e.g. a calibration shared by two models, or the
    overshootings of a model, shared by its backtests.
    """

    def __init__(self, spec: RunSpec) -> None:
        """Build the nodes of a run.

        Args:
            spec (RunSpec): The run spec.

        Raises:
            ValueError: If a value is out of a node's range, or an input is
                declared twice, differently.

        """
        builder = Builder()
        self._models = Pool(
            {name: builder.var_model(model) for name, model in spec.models.items()}
        )
        self._backtests = {
            model_name: Pool(
                {
                    name: builder.backtest(backtest, model)
                    for name, backtest in spec.backtests.items()
                }
            )
            for model_name, model in spec.models.items()
        }
        self._input_names = builder.input_names

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
