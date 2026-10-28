import datetime

import numpy as np
import pytest

from tail_lib.backtests.kupiec import Kupiec
from tail_lib.backtests.kupiec import kupiec_pof
from tail_lib.backtests.overshootings import Overshootings
from tail_lib.backtests.overshootings import overshootings
from tail_lib.context import Context
from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.value_at_risk import FilteredHistoricalSimulationsModel
from tail_lib.models.value_at_risk import GaussianModel
from tail_lib.models.value_at_risk import HistoricalSimulationsModel
from tail_lib.models.value_at_risk import ValueAtRiskInput
from tail_lib.models.volatility import ConstantVolatility
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.models.volatility import VolatilityFilter
from tail_lib.run import RunSpec
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import Trailing
from tests.helpers import returns_series


def _constant_context() -> Context:
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
    return ctx


def test_bounds_are_inclusive_and_every_access_uses_the_same_masked_var() -> None:
    returns = returns_series(n=80, n_assets=2)
    with Context() as ctx:
        vol = ConstantVolatility(0.01)
        model = GaussianModel(cf=0.99, vol=vol, name="gauss")
        GaussianModel(cf=0.99, vol=vol, name="alias")
    complete = ctx.run(returns)

    result = ctx.run(returns, lower=returns.dates[20], upper=returns.dates[40])

    assert result.returns is returns
    assert result[model] is result["gauss"] is result.models["alias"]
    np.testing.assert_array_equal(result[model].dates, returns.dates)
    np.testing.assert_array_equal(
        result[model].values[20:41], complete[model].values[20:41]
    )
    assert np.isnan(result[model].values[:20]).all()
    assert np.isnan(result[model].values[41:]).all()
    np.testing.assert_array_equal(result[vol].values, complete[vol].values)


def test_changing_the_range_never_moves_calibration_dates_or_refits_today() -> None:
    returns = returns_series(n=100)
    today_index = 75
    with Context() as ctx:
        decay = EwmaDecayCalibration(Trailing(20, step=13), start=Backcast(5), n_iter=4)
        vol = EwmaVolatility(decay, start=Backcast(5))
        GaussianModel(cf=0.99, vol=vol, name="gauss")
    full = ctx.run(returns)

    ranged = ctx.run(returns, lower=returns.dates[40], upper=returns.dates[75])
    today = ctx.run(returns, lower=returns.dates[75], upper=returns.dates[75])

    for result in (ranged, today):
        np.testing.assert_array_equal(result[decay].refits, full[decay].refits)
        np.testing.assert_array_equal(
            result[decay].values.values, full[decay].values.values
        )
        np.testing.assert_array_equal(result[vol].values, full[vol].values)
        assert result["gauss"].values[75] == full["gauss"].values[75]
    assert today_index not in today[decay].refits
    np.testing.assert_array_equal(
        ranged["gauss"].values[40:76], full["gauss"].values[40:76]
    )


def test_inputs_are_trimmed_before_selection_and_never_modified() -> None:
    returns = returns_series(n=80, n_assets=2)
    returns.values[:5, 1] = np.nan
    external = TimeSeries(
        dates=returns.dates, values=np.full(returns.values.shape, -0.02)
    )
    with Context() as ctx:
        var = ValueAtRiskInput("external", cf=0.99)

    result = ctx.run(
        returns,
        {"external": external},
        lower=returns.dates[20],
        upper=returns.dates[30],
    )

    np.testing.assert_array_equal(result.returns.dates, returns.dates[5:])
    np.testing.assert_array_equal(result[var].values[15:26], external.values[20:31])
    assert np.isnan(result[var].values[:15]).all()
    assert np.isnan(result[var].values[26:]).all()
    np.testing.assert_array_equal(external.values, -0.02)


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_omitted_bound_uses_the_history_endpoint(bound: str) -> None:
    returns = returns_series(n=30)
    ctx = _constant_context()
    date = returns.dates[10].astype(datetime.date)
    result = ctx.run(returns, **{bound: date})

    selected = slice(10, None) if bound == "lower" else slice(None, 11)
    assert np.isfinite(result["gauss"].values[selected]).all()
    assert np.isfinite(result["gauss"].values).sum() == returns.dates[selected].size


def test_bounds_between_observations_select_only_dates_inside_them() -> None:
    returns = TimeSeries(
        dates=np.array(
            ["2020-01-03", "2020-01-06", "2020-01-07"], dtype="datetime64[D]"
        ),
        values=np.array([0.01, 0.02, 0.03]),
    )
    result = _constant_context().run(
        returns, lower=datetime.date(2020, 1, 4), upper=datetime.date(2020, 1, 6)
    )

    np.testing.assert_array_equal(
        np.isfinite(result["gauss"].values), [False, True, False]
    )
    assert result.lower == returns.dates[1]
    assert result.upper == returns.dates[1]


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        ("2020-01-06", "2020-01-03", "after upper"),
        ("2019-12-31", "2020-01-03", "outside the trimmed history"),
        ("2020-01-03", "2021-01-01", "outside the trimmed history"),
        ("NaT", "2020-01-03", "cannot be NaT"),
        ("2020-01-04", "2020-01-05", "no observations"),
    ],
)
def test_invalid_bounds_fail(lower: str, upper: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _constant_context().run(
            returns_series(n=30),
            lower=np.datetime64(lower, "D"),
            upper=np.datetime64(upper, "D"),
        )


@pytest.mark.parametrize("dates", [["2020-01-02", "2020-01-01"], ["2020-01-01"] * 2])
def test_bounded_runs_require_increasing_unique_dates(dates: list[str]) -> None:
    returns = TimeSeries(
        dates=np.array(dates, dtype="datetime64[D]"), values=np.zeros(2)
    )
    with pytest.raises(ValueError, match="unique, increasing"):
        _constant_context().run(returns, lower=np.datetime64("2020-01-01"))


@pytest.mark.parametrize("kind", ["hs", "fhs", "external"])
def test_insufficient_history_is_reported_before_any_node_computes(kind: str) -> None:
    class _Never(ConstantVolatility):
        def compute(self, returns: TimeSeries, /) -> TimeSeries:  # noqa: ARG002
            pytest.fail("Preflight should fail before computing")

    returns = returns_series(n=80)
    inputs = {}
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_Never(0.01), name="first")
        if kind == "hs":
            HistoricalSimulationsModel(cf=0.99, window=Trailing(30), name="late")
        elif kind == "fhs":
            vol = EwmaVolatility(ConstantDecay(0.94), start=Backcast(10))
            FilteredHistoricalSimulationsModel(
                cf=0.99, filter=VolatilityFilter(vol), window=Trailing(15), name="late"
            )
        else:
            ValueAtRiskInput("late", cf=0.99)
            values = np.full(returns.values.shape, -0.02)
            values[:30] = np.nan
            inputs["late"] = TimeSeries(dates=returns.dates, values=values)

    with pytest.raises(ValueError, match="Model 'late' cannot forecast from"):
        ctx.run(returns, inputs, lower=returns.dates[20], upper=returns.dates[40])


def test_fhs_can_start_exactly_when_its_last_warm_up_finishes() -> None:
    returns = returns_series(n=80)
    with Context() as ctx:
        vol = EwmaVolatility(ConstantDecay(0.94), start=Backcast(10))
        FilteredHistoricalSimulationsModel(
            cf=0.99, filter=VolatilityFilter(vol), window=Trailing(15), name="fhs"
        )

    result = ctx.run(returns, lower=returns.dates[24], upper=returns.dates[24])
    assert np.isfinite(result["fhs"].values[24])
    assert np.isfinite(result["fhs"].values).sum() == 1


def test_backtests_use_only_flags_inside_the_range_and_end_at_upper() -> None:
    returns = returns_series(n=80, n_assets=2)
    with Context() as ctx:
        model = GaussianModel(cf=0.99, vol=ConstantVolatility(0.01), name="gauss")
        Kupiec(name="whole")
        Kupiec(name="rolling", window=Trailing(7, step=3))
        Kupiec(name="last", window=Last(10))
    full = ctx.run(returns)
    expected_flags = overshootings(full[model], returns)[21:51]

    result = ctx.run(returns, lower=returns.dates[20], upper=returns.dates[50])

    flags = result[Overshootings(model)]
    assert np.isnan(flags.values[:21]).all()
    assert np.isnan(flags.values[51:]).all()
    np.testing.assert_array_equal(flags.values[21:51], expected_flags.values)
    for name, window in (
        ("whole", None),
        ("rolling", Trailing(7, step=3)),
        ("last", Last(10)),
    ):
        expected = kupiec_pof(expected_flags, cf_var=0.99, window=window)
        actual = result.backtests["gauss"][name]
        np.testing.assert_array_equal(actual.end_dates, expected.end_dates)
        np.testing.assert_array_equal(actual.n_obs, expected.n_obs)
        np.testing.assert_array_equal(actual.p_value, expected.p_value)


def test_a_single_forecast_has_no_observations_to_backtest() -> None:
    returns = returns_series(n=30)
    ctx = _constant_context()
    ctx.add_backtest("whole", Kupiec())

    with pytest.raises(ValueError, match="at least two forecast dates"):
        ctx.run(returns, lower=returns.dates[20], upper=returns.dates[20])


def test_sparse_var_steps_are_rejected_only_for_bounded_runs() -> None:
    returns = returns_series(n=80)
    with Context() as ctx:
        HistoricalSimulationsModel(cf=0.99, window=Trailing(10, step=3), name="hs")
    ctx.run(returns)

    with pytest.raises(ValueError, match=r"models\.hs\.window.*step=1"):
        ctx.run(returns, lower=returns.dates[20])


def test_missing_forecasts_inside_the_range_are_reported() -> None:
    returns = returns_series(n=80)
    external = TimeSeries(
        dates=returns.dates, values=np.full(returns.values.shape, -0.02)
    )
    external.values[25] = np.nan
    with Context() as ctx:
        ValueAtRiskInput("external", cf=0.99)

    with pytest.raises(ValueError, match=r"external.*missing VaR"):
        ctx.run(returns, {"external": external}, lower=returns.dates[20])


def test_last_window_remains_a_forecast_on_the_final_history_date() -> None:
    returns = returns_series(n=80)
    with Context() as ctx:
        HistoricalSimulationsModel(cf=0.99, window=Last(10), name="last")

    result = ctx.run(returns, lower=returns.dates[-1], upper=returns.dates[-1])
    assert np.isfinite(result["last"].values[-1])
    assert np.isnan(result["last"].values[:-1]).all()
    with pytest.raises(ValueError, match="Model 'last' cannot forecast from"):
        ctx.run(returns, lower=returns.dates[-2], upper=returns.dates[-1])


def test_warm_up_checks_use_trimmed_history_not_the_original_prefix() -> None:
    returns = returns_series(n=80, n_assets=2)
    returns.values[:20, 1] = np.nan
    with Context() as ctx:
        HistoricalSimulationsModel(cf=0.99, window=Trailing(30), name="hs")

    with pytest.raises(ValueError, match="Model 'hs' cannot forecast from"):
        ctx.run(returns, lower=returns.dates[40])


@pytest.mark.parametrize("kind", ["volatility", "calibration"])
def test_invalid_ewma_initialization_fails_in_preflight(kind: str) -> None:
    class _Never(ConstantVolatility):
        def compute(self, returns: TimeSeries, /) -> TimeSeries:  # noqa: ARG002
            pytest.fail("Invalid EWMA history should be caught before computing")

    returns = returns_series(n=30)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_Never(0.01), name="first")
        if kind == "volatility":
            vol = EwmaVolatility(ConstantDecay(0.94), start=Backcast(40))
        else:
            vol = EwmaVolatility(EwmaDecayCalibration(Trailing(10)))
        GaussianModel(cf=0.99, vol=vol, name="invalid")

    with pytest.raises(ValueError, match=r"models\.invalid.*backcast"):
        ctx.run(returns, lower=returns.dates[20])


def test_unknown_custom_warm_up_is_checked_on_the_actual_forecast() -> None:
    class _Custom(ConstantVolatility):
        def compute(self, returns: TimeSeries, /) -> TimeSeries:
            result = super().compute(returns)
            result.values[:25] = np.nan
            return result

    returns = returns_series(n=50)
    with Context() as ctx:
        GaussianModel(cf=0.99, vol=_Custom(0.01), name="custom")

    with pytest.raises(ValueError, match=r"custom.*missing VaR"):
        ctx.run(returns, lower=returns.dates[20])


def test_a_context_from_specs_accepts_different_bounds_on_each_execution() -> None:
    returns = returns_series(n=80)
    ctx = Context.from_spec(
        RunSpec.model_validate(
            {
                "models": {
                    "hs": {
                        "type": "historical",
                        "cf": 0.99,
                        "window": {"type": "trailing", "length": 10},
                    }
                }
            }
        )
    )

    early = ctx.run(returns, lower=returns.dates[20], upper=returns.dates[40])
    late = ctx.run(returns, lower=returns.dates[30], upper=returns.dates[50])

    np.testing.assert_array_equal(early["hs"].values[30:41], late["hs"].values[30:41])
    assert np.isnan(early["hs"].values[41:]).all()
    assert np.isnan(late["hs"].values[:30]).all()


def test_effective_bounds_belong_to_the_results_of_each_execution() -> None:
    returns = returns_series(n=80)
    ctx = _constant_context()

    lower_only = ctx.run(returns, lower=returns.dates[20])
    upper_only = ctx.run(returns, upper=returns.dates[40])
    unbounded = ctx.run(returns)

    assert lower_only.lower == returns.dates[20]
    assert lower_only.upper == returns.dates[-1]
    assert upper_only.lower == returns.dates[0]
    assert upper_only.upper == returns.dates[40]
    assert unbounded.lower is None
    assert unbounded.upper is None
