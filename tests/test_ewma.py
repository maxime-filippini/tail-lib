import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from tail_lib.models.ewma import Backcast
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import DecayFactors
from tail_lib.models.ewma import EwmaDecayCalibration
from tail_lib.models.ewma import StartingVol
from tail_lib.models.ewma import fit_ewma_decay
from tail_lib.time_series.time_series import TimeSeries
from tail_lib.time_series.windows import Last
from tail_lib.time_series.windows import Trailing
from tests.helpers import run_node
from tests.helpers import simulate_ewma
from tests.helpers import simulated_returns


@pytest.mark.parametrize("window", [0, -1])
def test_backcast_rejects_empty_window(window: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Backcast(window=window)


def test_starting_vol_rejects_non_positive_vol() -> None:
    with pytest.raises(ValueError, match="positive"):
        StartingVol(np.array([0.01, 0.0]))


def test_constant_decay_rejects_decay_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        ConstantDecay(1.1)


def test_fit_ewma_decay_recovers_decay_per_batch_element() -> None:
    true = np.array([[0.9, 0.94, 0.97], [0.92, 0.96, 0.98]])
    r = simulate_ewma(true, n=2000)

    fitted = fit_ewma_decay(r)

    assert fitted.shape == true.shape
    np.testing.assert_allclose(fitted, true, atol=0.02)


def test_fit_ewma_decay_batched_matches_individual_fits() -> None:
    r = simulate_ewma(np.array([0.9, 0.97]), n=300)

    fitted = fit_ewma_decay(r)

    for i in range(r.shape[1]):
        np.testing.assert_allclose(fitted[i], fit_ewma_decay(r[:, i]))


def test_fit_ewma_decay_on_rolling_windows() -> None:
    r = simulate_ewma(np.array([0.9, 0.97]), n=300)
    w = 250
    windows = np.moveaxis(sliding_window_view(r, w, axis=0), -1, 0)

    fitted = fit_ewma_decay(windows)

    assert fitted.shape == (r.shape[0] - w + 1, r.shape[1])
    np.testing.assert_allclose(fitted[-1], fit_ewma_decay(r[-w:]))


@pytest.mark.parametrize(
    ("bounds", "window"),
    [((0.9, 0.8), 20), ((0.5, 1.1), 20), ((0.8, 0.99), 300)],
)
def test_fit_ewma_decay_rejects_invalid_arguments(
    bounds: tuple[float, float], window: int
) -> None:
    r = simulate_ewma(np.array([0.94]), n=300)

    with pytest.raises(ValueError, match=r"(bounds|Backcast)"):
        fit_ewma_decay(r, bounds=bounds, start=Backcast(window=window))


def test_fit_ewma_decay_with_starting_vol_uses_every_date() -> None:
    r = simulate_ewma(np.array([0.9, 0.97]), n=300)

    fitted = fit_ewma_decay(r, start=StartingVol(0.01))

    assert fitted.shape == (2,)
    assert not np.allclose(fitted, fit_ewma_decay(r))


def test_decay_calibration_fits_trailing_windows() -> None:
    returns = simulated_returns()
    r = returns.values

    decay = run_node(
        EwmaDecayCalibration(Trailing(length=100, step=20)), returns
    ).values

    np.testing.assert_array_equal(decay.dates, returns.dates)
    assert np.isnan(decay.values[:99]).all()
    np.testing.assert_allclose(decay.values[99], fit_ewma_decay(r[:100]))
    np.testing.assert_allclose(decay.values[99:119], np.tile(decay.values[99], (20, 1)))
    np.testing.assert_allclose(decay.values[119], fit_ewma_decay(r[20:120]))
    np.testing.assert_allclose(decay.values[159], fit_ewma_decay(r[60:160]))


def test_decay_calibration_rejects_window_longer_than_returns() -> None:
    with pytest.raises(ValueError, match="longer"):
        run_node(EwmaDecayCalibration(Trailing(length=200)), simulated_returns())


def test_calibration_on_last_window_fits_once_on_the_last_dates() -> None:
    returns = simulated_returns(n=160)

    last = run_node(EwmaDecayCalibration(Last(length=100)), returns).values
    rolling = run_node(EwmaDecayCalibration(Trailing(length=100)), returns).values

    assert np.isnan(last.values[:-1]).all()
    np.testing.assert_array_equal(last.values[-1], rolling.values[-1])


def test_a_constant_decay_has_one_refit_fitted_from_the_first_date() -> None:
    returns = simulated_returns()

    decay = run_node(ConstantDecay(np.array([0.9, 0.97])), returns)

    np.testing.assert_array_equal(decay.refits, [0])
    np.testing.assert_array_equal(decay.fit_starts, [0])
    np.testing.assert_array_equal(decay.values.values[-1], [0.9, 0.97])


def test_a_calibration_refits_on_each_window_end_fitted_from_its_start() -> None:
    returns = simulated_returns(n=160)

    rolling = run_node(EwmaDecayCalibration(Trailing(length=100, step=20)), returns)
    last = run_node(EwmaDecayCalibration(Last(length=100)), returns)

    np.testing.assert_array_equal(rolling.refits, [99, 119, 139, 159])
    np.testing.assert_array_equal(rolling.fit_starts, [0, 20, 40, 60])
    np.testing.assert_array_equal(last.refits, [159])
    np.testing.assert_array_equal(last.fit_starts, [60])


@pytest.mark.parametrize(
    ("refits", "fit_starts", "match"),
    [
        ([0, 5], [0], "one index per refit"),
        ([5, 5], [0, 0], "increasing dates"),
        ([10], [0], "increasing dates of the 10 dates"),
        ([5], [6], "on or before its refit"),
    ],
)
def test_decay_factors_check_their_refits(
    refits: list[int], fit_starts: list[int], match: str
) -> None:
    values = simulated_returns(n=10)
    values = TimeSeries(dates=values.dates, values=np.full(values.values.shape, 0.9))

    with pytest.raises(ValueError, match=match):
        DecayFactors(
            values=values, refits=np.array(refits), fit_starts=np.array(fit_starts)
        )
