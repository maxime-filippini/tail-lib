"""Build nodes from specs (R9a).

A `Builder` turns each spec into its node, building the node's inputs from
the nested specs first. It remembers the node built for each spec, so
identical specs, even written separately, give one node instance, which a
`Cache` then computes once (D3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import assert_never

import numpy as np

from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import EwmaStart
from tail_lib.models.ewma import StartingVol
from tail_lib.models.mean import ConstantMean
from tail_lib.models.mean import MeanModel
from tail_lib.models.mean import ZeroMean
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.value_at_risk import ValueAtRiskModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import FittedEwmaVolatility
from tail_lib.models.volatility import VolatilityInput
from tail_lib.models.volatility import VolatilityModel
from tail_lib.specs import BackcastSpec
from tail_lib.specs import ConstantDecaySpec
from tail_lib.specs import ConstantMeanSpec
from tail_lib.specs import ConstantVolatilitySpec
from tail_lib.specs import EwmaDecayCalibrationSpec
from tail_lib.specs import EwmaStartSpec
from tail_lib.specs import EwmaVolatilitySpec
from tail_lib.specs import FittedEwmaVolatilitySpec
from tail_lib.specs import GaussianSpec
from tail_lib.specs import HistoricalSimulationsSpec
from tail_lib.specs import MeanSpec
from tail_lib.specs import StartingVolSpec
from tail_lib.specs import TrailingSpec
from tail_lib.specs import ValueAtRiskInputSpec
from tail_lib.specs import ValueAtRiskSpec
from tail_lib.specs import Values
from tail_lib.specs import VolatilityInputSpec
from tail_lib.specs import VolatilitySpec
from tail_lib.specs import ZeroMeanSpec
from tail_lib.windows import Trailing

if TYPE_CHECKING:
    from collections.abc import Callable

    from tail_lib._typing import FloatArray


class Builder:
    """Builds nodes from specs, one node per distinct spec.

    Use one builder for all the specs of a run, so their identical parts are
    shared. There is one method per node role, e.g. `volatility`, each typed
    with the specs it accepts and the node it returns.
    """

    def __init__(self) -> None:
        # One table per role keeps each lookup typed: spec in, node out.
        self._var_models: dict[ValueAtRiskSpec, ValueAtRiskModel] = {}
        self._volatilities: dict[VolatilitySpec, VolatilityModel] = {}
        self._means: dict[MeanSpec, MeanModel] = {}
        self._constant_decays: dict[ConstantDecaySpec, ConstantDecay] = {}
        self._calibrations: dict[EwmaDecayCalibrationSpec, EwmaDecayCalibration] = {}

    def var_model(self, spec: ValueAtRiskSpec) -> ValueAtRiskModel:
        """Get the value-at-risk model of a spec, building it on first request."""
        return _get_or_build(self._var_models, spec, self._build_var_model)

    def volatility(self, spec: VolatilitySpec) -> VolatilityModel:
        """Get the volatility model of a spec, building it on first request."""
        return _get_or_build(self._volatilities, spec, self._build_volatility)

    def mean(self, spec: MeanSpec) -> MeanModel:
        """Get the mean model of a spec, building it on first request."""
        return _get_or_build(self._means, spec, self._build_mean)

    def constant_decay(self, spec: ConstantDecaySpec) -> ConstantDecay:
        """Get the constant decay factor of a spec, building it on first request."""
        return _get_or_build(self._constant_decays, spec, _build_constant_decay)

    def calibration(self, spec: EwmaDecayCalibrationSpec) -> EwmaDecayCalibration:
        """Get the decay calibration of a spec, building it on first request."""
        return _get_or_build(self._calibrations, spec, _build_calibration)

    def _build_var_model(self, spec: ValueAtRiskSpec) -> ValueAtRiskModel:
        match spec:
            case ValueAtRiskInputSpec():
                return ValueAtRiskInput(spec.name, cf=spec.cf)
            case HistoricalSimulationsSpec():
                return HistoricalSimulationsModel(
                    cf=spec.cf,
                    window=_trailing(spec.window),
                    interpolation=spec.interpolation,
                )
            case GaussianSpec():
                return GaussianModel(
                    cf=spec.cf,
                    vol_model=self.volatility(spec.vol),
                    mean_model=self.mean(spec.mean),
                )
            case _:
                assert_never(spec)

    def _build_volatility(self, spec: VolatilitySpec) -> VolatilityModel:
        match spec:
            case VolatilityInputSpec():
                return VolatilityInput(spec.name)
            case ConstantVolatilitySpec():
                return ConstantVolatility(_values(spec.vol))
            case EwmaVolatilitySpec():
                return EwmaVolatility(
                    self.constant_decay(spec.decay), start=_ewma_start(spec.start)
                )
            case FittedEwmaVolatilitySpec():
                return FittedEwmaVolatility(self.calibration(spec.calibration))
            case _:
                assert_never(spec)

    def _build_mean(self, spec: MeanSpec) -> MeanModel:
        match spec:
            case ZeroMeanSpec():
                return ZeroMean()
            case ConstantMeanSpec():
                return ConstantMean(_values(spec.mean))
            case _:
                assert_never(spec)


def _get_or_build[S, N](built: dict[S, N], spec: S, build: Callable[[S], N]) -> N:
    if spec not in built:
        built[spec] = build(spec)
    return built[spec]


def _build_constant_decay(spec: ConstantDecaySpec) -> ConstantDecay:
    return ConstantDecay(_values(spec.value))


def _build_calibration(spec: EwmaDecayCalibrationSpec) -> EwmaDecayCalibration:
    return EwmaDecayCalibration(
        _trailing(spec.window),
        bounds=spec.bounds,
        start=_ewma_start(spec.start),
        n_iter=spec.n_iter,
    )


def _trailing(spec: TrailingSpec) -> Trailing:
    return Trailing(length=spec.length, step=spec.step)


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
