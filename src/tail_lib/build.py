"""Build nodes from specs (R9a).

Contexts use the plain `_build_*` functions: equal nodes share computation
by value (D9), without tables of nodes built from specs. `_check_input_specs`
rejects conflicting declarations before building nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import assert_never

import numpy as np

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.backtests.kupiec import KupiecResult
from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import DecayModel
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import EwmaStart
from tail_lib.models.ewma import StartingVol
from tail_lib.models.mean import ConstantMean
from tail_lib.models.mean import MeanModel
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.models.volatility import VolatilityInput
from tail_lib.models.volatility import VolatilityModel
from tail_lib.specs import BackcastSpec
from tail_lib.specs import BacktestSpec
from tail_lib.specs import ConstantDecaySpec
from tail_lib.specs import ConstantMeanSpec
from tail_lib.specs import ConstantVolatilitySpec
from tail_lib.specs import DecaySpec
from tail_lib.specs import EwmaDecayCalibrationSpec
from tail_lib.specs import EwmaStartSpec
from tail_lib.specs import EwmaVolatilitySpec
from tail_lib.specs import FilteredHistoricalSimulationsSpec
from tail_lib.specs import GaussianSpec
from tail_lib.specs import HistoricalSimulationsSpec
from tail_lib.specs import KupiecSpec
from tail_lib.specs import LastSpec
from tail_lib.specs import MeanSpec
from tail_lib.specs import Spec
from tail_lib.specs import StartingVolSpec
from tail_lib.specs import TrailingSpec
from tail_lib.specs import ValueAtRiskInputSpec
from tail_lib.specs import ValueAtRiskSpec
from tail_lib.specs import Values
from tail_lib.specs import VolatilityFilterSpec
from tail_lib.specs import VolatilityInputSpec
from tail_lib.specs import VolatilitySpec
from tail_lib.specs import WindowSpec
from tail_lib.specs import ZeroMeanSpec
from tail_lib.windows import Last
from tail_lib.windows import Trailing

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tail_lib._typing import FloatArray

# A union, like the specs, once there is a second backtest.
type BacktestResult = KupiecResult


def _check_input_specs(specs: Iterable[Spec]) -> None:
    """Reject two different specs declaring the same run input (R13).

    This table checks declarations only; node sharing is by value (D9).
    """
    declared: dict[str, ValueAtRiskInputSpec | VolatilityInputSpec] = {}
    pending = list(specs)
    while pending:
        spec = pending.pop()
        if isinstance(spec, ValueAtRiskInputSpec | VolatilityInputSpec):
            first = declared.setdefault(spec.name, spec)
            if first != spec:
                msg = (
                    f"Input {spec.name!r} is declared twice, differently: "
                    f"{first!r} and {spec!r}!"
                )
                raise ValueError(msg)
        pending.extend(value for _, value in spec if isinstance(value, Spec))


def _build_var_model(spec: ValueAtRiskSpec, *, name: str) -> ValueAtRiskModel:
    """Build a named VaR model and its inputs, without sharing tables."""
    match spec:
        case ValueAtRiskInputSpec():
            if name != spec.name:
                msg = (
                    f"Input model {name!r}, at `models.{name}`, reads "
                    f"{spec.name!r}: its model key has to match its input name!"
                )
                raise ValueError(msg)
            return ValueAtRiskInput(spec.name, cf=spec.cf)
        case HistoricalSimulationsSpec():
            return HistoricalSimulationsModel(
                cf=spec.cf,
                window=_window(spec.window),
                interpolation=spec.interpolation,
                name=name,
            )
        case FilteredHistoricalSimulationsSpec():
            return FilteredHistoricalSimulationsModel(
                cf=spec.cf,
                filter=_build_volatility_filter(spec.filter),
                window=_window(spec.window),
                interpolation=spec.interpolation,
                name=name,
            )
        case GaussianSpec():
            return GaussianModel(
                cf=spec.cf,
                vol=_build_volatility(spec.vol),
                mean=_build_mean(spec.mean),
                name=name,
            )
        case _:
            assert_never(spec)


def _build_volatility(spec: VolatilitySpec) -> VolatilityModel:
    """Build a volatility model and its inputs."""
    match spec:
        case VolatilityInputSpec():
            return VolatilityInput(spec.name)
        case ConstantVolatilitySpec():
            return ConstantVolatility(_values(spec.vol))
        case EwmaVolatilitySpec():
            return EwmaVolatility(
                _build_decay(spec.decay), start=_ewma_start(spec.start)
            )
        case _:
            assert_never(spec)


def _build_decay(spec: DecaySpec) -> DecayModel:
    """Build a constant decay or a calibration."""
    match spec:
        case ConstantDecaySpec():
            return _build_constant_decay(spec)
        case EwmaDecayCalibrationSpec():
            return _build_calibration(spec)
        case _:
            assert_never(spec)


def _build_volatility_filter(spec: VolatilityFilterSpec) -> VolatilityFilter:
    """Build a filter with the volatility and mean of its spec."""
    return VolatilityFilter(_build_volatility(spec.vol), mean=_build_mean(spec.mean))


def _build_mean(spec: MeanSpec) -> MeanModel:
    """Build a mean model."""
    match spec:
        case ZeroMeanSpec():
            return ZeroMean()
        case ConstantMeanSpec():
            return ConstantMean(_values(spec.mean))
        case _:
            assert_never(spec)


def _build_backtest(spec: BacktestSpec) -> Kupiec:
    """Build a declaration; the context supplies each model when binding."""
    match spec:
        case KupiecSpec():
            window = None if spec.window is None else _window(spec.window)
            return Kupiec(window=window)
        case _:
            assert_never(spec)


def _build_constant_decay(spec: ConstantDecaySpec) -> ConstantDecay:
    return ConstantDecay(_values(spec.value))


def _build_calibration(spec: EwmaDecayCalibrationSpec) -> EwmaDecayCalibration:
    return EwmaDecayCalibration(
        _window(spec.window),
        bounds=spec.bounds,
        start=_ewma_start(spec.start),
        n_iter=spec.n_iter,
    )


def _window(spec: WindowSpec) -> Trailing | Last:
    match spec:
        case TrailingSpec():
            return Trailing(length=spec.length, step=spec.step)
        case LastSpec():
            return Last(length=spec.length)
        case _:
            assert_never(spec)


def _ewma_start(spec: EwmaStartSpec) -> EwmaStart:
    match spec:
        case BackcastSpec():
            return Backcast(window=spec.window)
        case StartingVolSpec():
            return StartingVol(_values(spec.vol))
        case _:
            assert_never(spec)


def _values(values: Values[float]) -> FloatArray | float:
    """Turn the values of a spec into what nodes take: a float or an array."""
    if isinstance(values, tuple):
        return np.asarray(values, dtype=np.float64)
    return values
