"""The serialized spec of a context, parsed from JSON or Python data.

`Context.from_spec` builds its nodes. Computation and graph inspection live
in `tail_lib.context`; this module keeps the `RunSpec` import path.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field
from pydantic import model_validator

from tail_lib.context import _check_names_and_labels
from tail_lib.specs import BacktestSpec
from tail_lib.specs import Label
from tail_lib.specs import Name
from tail_lib.specs import Spec
from tail_lib.specs import ValueAtRiskSpec
from tail_lib.specs import find_last_window


class RunSpec(Spec):
    """Spec of a run: named models, and named backtests run on each of them.

    The run inputs are not listed here: they are the input specs used in the
    models, e.g. `{"type": "input", "name": "var_ext", "cf": 0.99}`.

    A model with a `last` window, or with an input that has one, gives one
    value-at-risk, the forecast for a date after the data. It has nothing to
    backtest, so a run with backtests rejects it.

    `labels` gives display names to models and backtests, e.g. for a report,
    as in `Context`.
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
