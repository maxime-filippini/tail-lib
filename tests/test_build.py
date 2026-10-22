import numpy as np
import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError

from tail_lib.build import Builder
from tail_lib.graph import Pool
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import FittedEwmaVolatility
from tail_lib.specs import EwmaDecayCalibrationSpec
from tail_lib.specs import GaussianSpec
from tail_lib.specs import TrailingSpec
from tail_lib.specs import ValueAtRiskSpec
from tail_lib.time_series import TimeSeries
from tail_lib.windows import Trailing
from tests.helpers import returns_series
from tests.helpers import simulated_returns

_VAR_SPEC: TypeAdapter[ValueAtRiskSpec] = TypeAdapter(ValueAtRiskSpec)

_FITTED_EWMA = {
    "type": "fitted_ewma",
    "calibration": {"window": {"length": 100, "step": 50}},
}


def test_gaussian_spec_builds_the_same_model_as_python() -> None:
    returns = simulated_returns(n=300)
    spec = _VAR_SPEC.validate_python(
        {"type": "gaussian", "cf": 0.99, "vol": _FITTED_EWMA}
    )

    built = Builder().var_model(spec).run(returns)

    calibration = EwmaDecayCalibration(Trailing(length=100, step=50))
    expected = GaussianModel(cf=0.99, vol_model=FittedEwmaVolatility(calibration)).run(
        returns
    )
    np.testing.assert_array_equal(built.values, expected.values)


def test_historical_spec_builds_the_same_model_as_python() -> None:
    returns = returns_series(n=200)
    spec = _VAR_SPEC.validate_python(
        {"type": "historical", "cf": 0.95, "window": {"length": 50}}
    )

    built = Builder().var_model(spec).run(returns)

    expected = HistoricalSimulationsModel(cf=0.95, window=Trailing(length=50)).run(
        returns
    )
    np.testing.assert_array_equal(built.values, expected.values)


def test_identical_specs_written_separately_share_one_node() -> None:
    builder = Builder()
    spec_99 = GaussianSpec.model_validate({"cf": 0.99, "vol": _FITTED_EWMA})
    spec_95 = GaussianSpec.model_validate_json(
        '{"cf": 0.95, "vol": {"type": "fitted_ewma",'
        ' "calibration": {"window": {"length": 100, "step": 50}}}}'
    )

    model_99 = builder.var_model(spec_99)
    model_95 = builder.var_model(spec_95)

    assert model_99 is not model_95
    assert builder.var_model(spec_99) is model_99
    assert spec_99.vol == spec_95.vol
    assert builder.volatility(spec_99.vol) is builder.volatility(spec_95.vol)


def test_shared_calibration_is_fitted_once_in_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_fits = 0
    compute = EwmaDecayCalibration.compute

    def counting_compute(self: EwmaDecayCalibration, returns: TimeSeries) -> TimeSeries:
        nonlocal n_fits
        n_fits += 1
        return compute(self, returns)

    builder = Builder()
    models = {
        f"gauss_{cf}": builder.var_model(
            _VAR_SPEC.validate_python(
                {"type": "gaussian", "cf": cf, "vol": _FITTED_EWMA}
            )
        )
        for cf in (0.95, 0.99)
    }

    monkeypatch.setattr(EwmaDecayCalibration, "compute", counting_compute)
    Pool(models).run(simulated_returns(n=300))

    assert n_fits == 1


def test_list_values_become_one_value_per_series() -> None:
    returns = returns_series(n=50, n_assets=2)
    spec = _VAR_SPEC.validate_json(
        '{"type": "gaussian", "cf": 0.99,'
        ' "vol": {"type": "constant", "vol": [0.01, 0.02]}}'
    )

    built = Builder().var_model(spec).run(returns)

    expected = GaussianModel(
        cf=0.99, vol_model=ConstantVolatility(np.array([0.01, 0.02]))
    ).run(returns)
    np.testing.assert_array_equal(built.values, expected.values)


def test_spec_defaults_match_node_defaults() -> None:
    spec = EwmaDecayCalibrationSpec(window=TrailingSpec(length=100))

    built = Builder().calibration(spec)

    default = EwmaDecayCalibration(Trailing(length=100))
    assert built.window == default.window
    assert built.start == default.start


def test_unknown_type_is_reported_with_its_path() -> None:
    with pytest.raises(ValidationError, match=r"vol\n.*'ewmaa'"):
        _VAR_SPEC.validate_python(
            {"type": "gaussian", "cf": 0.99, "vol": {"type": "ewmaa"}}
        )


def test_unknown_field_is_an_error() -> None:
    with pytest.raises(ValidationError, match=r"gaussian\.cff"):
        _VAR_SPEC.validate_python(
            {"type": "gaussian", "cff": 0.99, "vol": {"type": "constant", "vol": 0.01}}
        )


_CONSTANT_VOL = {"type": "constant", "vol": 0.01}


@pytest.mark.parametrize(
    ("spec", "path"),
    [
        ({"type": "gaussian", "cf": 1.5, "vol": _CONSTANT_VOL}, r"gaussian\.cf\n"),
        ({"type": "gaussian", "cf": 0.0, "vol": _CONSTANT_VOL}, r"gaussian\.cf\n"),
        (
            {"type": "gaussian", "cf": 0.99, "vol": {"type": "constant", "vol": -0.01}},
            r"gaussian\.vol\.constant\.vol\.single\n",
        ),
        (
            {
                "type": "gaussian",
                "cf": 0.99,
                "vol": {"type": "ewma", "decay": {"value": [0.94, 1.2]}},
            },
            r"gaussian\.vol\.ewma\.decay\.value\.list\.1\n",
        ),
        (
            {"type": "gaussian", "cf": 0.99, "vol": {"type": "constant", "vol": []}},
            r"gaussian\.vol\.constant\.vol\.list\n",
        ),
        (
            {"type": "historical", "cf": 0.99, "window": {"length": 0}},
            r"historical\.window\.length\n",
        ),
        (
            {
                "type": "gaussian",
                "cf": 0.99,
                "vol": {
                    "type": "fitted_ewma",
                    "calibration": {
                        "window": {"length": 100},
                        "bounds": [0.99, 0.9],
                    },
                },
            },
            r"calibration\.bounds\n.*Lower bound has to be below upper bound",
        ),
        (
            {"type": "input", "name": "", "cf": 0.99},
            r"input\.name\n",
        ),
    ],
)
def test_out_of_range_values_are_reported_with_their_path(
    spec: dict[str, object], path: str
) -> None:
    with pytest.raises(ValidationError, match=path) as error:
        _VAR_SPEC.validate_python(spec)

    assert error.value.error_count() == 1


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        _VAR_SPEC.validate_json(
            '{"type": "gaussian", "cf": 0.99, "vol": {"type": "constant", "vol": NaN}}'
        )
