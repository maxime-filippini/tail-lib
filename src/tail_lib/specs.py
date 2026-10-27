"""Specs: models written as data, e.g. parsed from JSON (R9).

A spec describes a node and its settings, and nothing more: `build` turns
specs into nodes. Each spec class mirrors one node class, and its fields
mirror the node's constructor arguments. A node input is a nested spec, e.g.
the volatility model of a Gaussian model.

Specs are frozen and compare by value, so identical specs, even written
separately, can be built into one shared node (D3).

Specs check everything they are given: the field names, the `type` tag
choosing the spec class in each slot, and the range of each value, e.g. a
confidence level in (0, 1). An error gives the path to the faulty value. The
nodes check the same ranges again, for nodes built directly in Python.
"""

from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import field_validator
from pydantic import model_validator

from tail_lib._checks import check_name
from tail_lib.models.ewma import DEFAULT_DECAY_BOUNDS
from tail_lib.models.ewma import DEFAULT_FIT_ITERATIONS
from tail_lib.models.value_at_risk import InterpolationMethod  # noqa: TC001

# Value ranges. They match the nodes' checks, and add a few the nodes lack.

type ConfidenceLevel = Annotated[float, Field(gt=0, lt=1)]
"""A confidence level, in (0, 1)."""

type DecayFactor = Annotated[float, Field(ge=0, le=1)]
"""An EWMA decay factor, in [0, 1]."""

type Positive = Annotated[float, Field(gt=0)]
"""A positive number, e.g. a volatility."""

type Count = Annotated[int, Field(ge=1)]
"""A number of dates or iterations, at least 1."""


def _identifier(name: str) -> str:
    check_name("A", name)
    return name


type Name = Annotated[str, AfterValidator(_identifier)]
"""The name of a model, a backtest or a run input: a Python identifier."""

type Label = Annotated[str, Field(min_length=1)]
"""The display name of a model or a backtest, any non-empty text."""


def _shape(value: Any) -> str:  # noqa: ANN401 - pydantic passes raw input.
    return "list" if isinstance(value, list | tuple) else "single"


type Values[T] = Annotated[
    Annotated[T, Tag("single")]
    | Annotated[tuple[T, ...], Field(min_length=1), Tag("list")],
    Discriminator(_shape),
]
"""A single value, or one value per series (a non-empty JSON list).

The input's shape picks the branch, so an error only reports that branch, e.g.
`vol.list.1` for the second value of a list.
"""


class Spec(BaseModel):
    """Base class of specs: frozen, with no fields beyond the declared ones.

    Floats have to be finite: NaN and infinities are rejected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


# Windows and EWMA starts: settings, not nodes.


class TrailingSpec(Spec):
    """Spec of `Trailing` windows."""

    type: Literal["trailing"] = "trailing"
    length: Count
    step: Count = 1


class LastSpec(Spec):
    """Spec of a `Last` window."""

    type: Literal["last"] = "last"
    length: Count


type WindowSpec = Annotated[TrailingSpec | LastSpec, Field(discriminator="type")]
"""Trailing windows, or one window on the last dates only."""


def find_last_window(spec: Spec) -> str | None:
    """Find a `last` window in a spec or in the specs it holds.

    A model with a `last` window, or with an input that has one, gives one
    value, on the last date.

    Args:
        spec (Spec): The spec to search, e.g. a model's.

    Returns:
        str | None: The dotted path to the first `last` window found, e.g.
            `vol.calibration.window`, or `None` if there is none.

    """
    for name, value in spec:
        if isinstance(value, LastSpec):
            return name
        if isinstance(value, Spec):
            inner = find_last_window(value)
            if inner is not None:
                return f"{name}.{inner}"
    return None


class BackcastSpec(Spec):
    """Spec of a `Backcast` start of the EWMA recursion."""

    type: Literal["backcast"] = "backcast"
    window: Count = 20


class StartingVolSpec(Spec):
    """Spec of a `StartingVol` start of the EWMA recursion."""

    type: Literal["starting_vol"] = "starting_vol"
    vol: Values[Positive]


type EwmaStartSpec = Annotated[
    BackcastSpec | StartingVolSpec, Field(discriminator="type")
]


# Means.


class ZeroMeanSpec(Spec):
    """Spec of a `ZeroMean` model."""

    type: Literal["zero"] = "zero"


class ConstantMeanSpec(Spec):
    """Spec of a `ConstantMean` model."""

    type: Literal["constant"] = "constant"
    mean: Values[float]


type MeanSpec = Annotated[ZeroMeanSpec | ConstantMeanSpec, Field(discriminator="type")]


# Decay factors.


class ConstantDecaySpec(Spec):
    """Spec of a `ConstantDecay` model."""

    type: Literal["constant"] = "constant"
    value: Values[DecayFactor]


class EwmaDecayCalibrationSpec(Spec):
    """Spec of an `EwmaDecayCalibration`."""

    type: Literal["ewma_calibration"] = "ewma_calibration"
    window: WindowSpec
    bounds: tuple[DecayFactor, DecayFactor] = DEFAULT_DECAY_BOUNDS
    start: EwmaStartSpec = BackcastSpec()
    n_iter: Count = DEFAULT_FIT_ITERATIONS

    @field_validator("bounds")
    @classmethod
    def _check_bounds_order(cls, bounds: tuple[float, float]) -> tuple[float, float]:
        lower, upper = bounds
        if lower >= upper:
            msg = f"Lower bound has to be below upper bound, got {bounds}"
            raise ValueError(msg)
        return bounds


# Volatilities.


class VolatilityInputSpec(Spec):
    """Spec of a `VolatilityInput`."""

    type: Literal["input"] = "input"
    name: Name


class ConstantVolatilitySpec(Spec):
    """Spec of a `ConstantVolatility` model."""

    type: Literal["constant"] = "constant"
    vol: Values[Positive]


type DecaySpec = Annotated[
    ConstantDecaySpec | EwmaDecayCalibrationSpec, Field(discriminator="type")
]
"""A constant decay factor, or one calibrated on the returns."""


class EwmaVolatilitySpec(Spec):
    """Spec of an `EwmaVolatility` model.

    With a calibrated decay, `start` has to be the calibration's, as in the
    node: the volatility is re-filtered the way the decay was fitted.
    """

    type: Literal["ewma"] = "ewma"
    decay: DecaySpec
    start: EwmaStartSpec = BackcastSpec()

    @model_validator(mode="after")
    def _check_start_matches_calibration(self) -> EwmaVolatilitySpec:
        if isinstance(self.decay, EwmaDecayCalibrationSpec) and (
            self.decay.start != self.start
        ):
            msg = (
                f"`start` is {self.start!r}, but the calibration in `decay` "
                f"starts with {self.decay.start!r}: give both the same start"
            )
            raise ValueError(msg)
        return self


type VolatilitySpec = Annotated[
    VolatilityInputSpec | ConstantVolatilitySpec | EwmaVolatilitySpec,
    Field(discriminator="type"),
]


class VolatilityFilterSpec(Spec):
    """Spec of a `VolatilityFilter`: returns standardized by `mean` and `vol`."""

    type: Literal["volatility_filter"] = "volatility_filter"
    vol: VolatilitySpec
    mean: MeanSpec = ZeroMeanSpec()


# Value-at-risk models.


class ValueAtRiskInputSpec(Spec):
    """Spec of a `ValueAtRiskInput`."""

    type: Literal["input"] = "input"
    name: Name
    cf: ConfidenceLevel


class HistoricalSimulationsSpec(Spec):
    """Spec of a `HistoricalSimulationsModel`."""

    type: Literal["historical"] = "historical"
    cf: ConfidenceLevel
    window: WindowSpec
    interpolation: InterpolationMethod = "linear"


class FilteredHistoricalSimulationsSpec(Spec):
    """Spec of a `FilteredHistoricalSimulationsModel`."""

    type: Literal["filtered_historical"] = "filtered_historical"
    cf: ConfidenceLevel
    filter: VolatilityFilterSpec
    window: WindowSpec
    interpolation: InterpolationMethod = "linear"


class GaussianSpec(Spec):
    """Spec of a `GaussianModel`."""

    type: Literal["gaussian"] = "gaussian"
    cf: ConfidenceLevel
    vol: VolatilitySpec
    mean: MeanSpec = ZeroMeanSpec()


type ValueAtRiskSpec = Annotated[
    ValueAtRiskInputSpec
    | HistoricalSimulationsSpec
    | FilteredHistoricalSimulationsSpec
    | GaussianSpec,
    Field(discriminator="type"),
]


# Backtests.


class KupiecSpec(Spec):
    """Spec of a `KupiecTest`, run on the overshootings of a model.

    Without a `window`, the test runs on the whole sample.
    """

    type: Literal["kupiec"] = "kupiec"
    window: WindowSpec | None = None


# A tagged union, like the other roles, once there is a second backtest.
type BacktestSpec = KupiecSpec
