import numpy as np
import pytest

from tail_lib.calendar import Calendar
from tail_lib.distributions import StudentT
from tail_lib.models.ewma import ConstantDecay
from tail_lib.models.ewma import StartingVol
from tail_lib.models.volatility import EwmaVolatility
from tail_lib.simulation import ArProcess
from tail_lib.simulation import EwmaProcess
from tail_lib.simulation import GarchProcess
from tail_lib.simulation import Simulator
from tests.helpers import FIRST_DATE

GARCH = GarchProcess(omega=2e-6, alpha=0.08, beta=0.9)
GARCH_LONG_RUN_VAR = 2e-6 / (1 - 0.08 - 0.9)


def test_simulate_gives_dates_and_shapes() -> None:
    sim = Simulator(vol=GARCH).simulate(
        10, first_date=FIRST_DATE, batch_shape=(3, 2), rng=0
    )

    for series in (sim.returns, sim.means, sim.vols):
        assert series.values.shape == (10, 3, 2)
        np.testing.assert_array_equal(
            series.dates, Calendar().dates_from(FIRST_DATE, 10)
        )


def test_same_seed_gives_same_paths() -> None:
    simulator = Simulator(vol=GARCH, shocks=StudentT(df=5))

    first = simulator.simulate(50, first_date=FIRST_DATE, rng=7)
    second = simulator.simulate(50, first_date=FIRST_DATE, rng=7)
    other = simulator.simulate(50, first_date=FIRST_DATE, rng=8)

    np.testing.assert_array_equal(first.returns.values, second.returns.values)
    assert not np.array_equal(first.returns.values, other.returns.values)


def test_burn_in_drops_first_dates() -> None:
    simulator = Simulator(vol=GARCH, mean=ArProcess(const=1e-4, ar=(0.2,)))

    full = simulator.simulate(30, first_date=FIRST_DATE, rng=1)
    burnt = simulator.simulate(20, first_date=FIRST_DATE, burn_in=10, rng=1)

    np.testing.assert_array_equal(burnt.returns.values, full.returns.values[10:])
    np.testing.assert_array_equal(burnt.vols.values, full.vols.values[10:])
    np.testing.assert_array_equal(burnt.means.values, full.means.values[10:])
    assert burnt.returns.dates[0] == np.datetime64(FIRST_DATE)


def test_ewma_volatility_filters_back_simulated_vols() -> None:
    # Same recursion and timing on both sides: the model given the true decay
    # and starting volatility gives the simulated volatilities exactly.
    decay = np.array([0.9, 0.94, 0.97])
    sim = Simulator(vol=EwmaProcess(decay=decay, start_vol=0.01)).simulate(
        200, first_date=FIRST_DATE, batch_shape=(3,), rng=2
    )

    model = EwmaVolatility(ConstantDecay(decay), start=StartingVol(0.01))

    np.testing.assert_allclose(
        model.run(sim.returns).values, sim.vols.values, rtol=1e-12
    )


def test_garch_vols_follow_recursion_on_residuals() -> None:
    mean = ArProcess(const=1e-4, ar=(0.3, -0.1))
    sim = Simulator(vol=GARCH, mean=mean).simulate(
        100, first_date=FIRST_DATE, batch_shape=(2,), rng=3
    )
    r = sim.returns.values
    long_run = mean.long_run_mean()

    # The mean and variance of date 0, then each date's from the one before.
    mu = np.full(r.shape, np.nan)
    mu[0] = 1e-4 + 0.3 * long_run - 0.1 * long_run
    mu[1:] = sim.means.values[:-1]
    var = np.full(r.shape, GARCH_LONG_RUN_VAR)
    var[1:] = sim.vols.values[:-1] ** 2
    eps = r - mu
    expected = np.sqrt(2e-6 + 0.08 * eps**2 + 0.9 * var)

    np.testing.assert_allclose(sim.vols.values, expected, rtol=1e-12)


def test_ar_means_use_past_returns() -> None:
    sim = Simulator(vol=GARCH, mean=ArProcess(const=1e-4, ar=(0.3, -0.1))).simulate(
        50, first_date=FIRST_DATE, rng=4
    )
    r = sim.returns.values

    # The mean on date `t` is the forecast for `t + 1`.
    np.testing.assert_allclose(
        sim.means.values[1:], 1e-4 + 0.3 * r[1:] - 0.1 * r[:-1], rtol=1e-12
    )


def test_garch_recovers_long_run_variance() -> None:
    sim = Simulator(vol=GARCH).simulate(
        500, first_date=FIRST_DATE, batch_shape=(400,), rng=5
    )

    np.testing.assert_allclose(
        np.mean(sim.returns.values**2), GARCH_LONG_RUN_VAR, rtol=0.05
    )


def test_ar_recovers_long_run_mean_and_autocorrelation() -> None:
    mean = ArProcess(const=2e-3, ar=(0.5,))
    sim = Simulator(vol=GARCH, mean=mean).simulate(
        2000, first_date=FIRST_DATE, batch_shape=(50,), rng=6
    )
    r = sim.returns.values
    centered = r - r.mean(axis=0)
    autocorrelation = np.mean(centered[1:] * centered[:-1]) / np.mean(centered**2)

    np.testing.assert_allclose(r.mean(), 4e-3, rtol=0.05)
    np.testing.assert_allclose(autocorrelation, 0.5, atol=0.02)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"omega": 0.0, "alpha": 0.1, "beta": 0.8}, "omega"),
        ({"omega": 1e-6, "alpha": -0.1, "beta": 0.8}, "at least 0"),
        ({"omega": 1e-6, "alpha": 0.1, "beta": np.array([0.8, 0.9])}, "below 1"),
    ],
)
def test_garch_rejects_invalid_parameters(params: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GarchProcess(**params)


def test_garch_from_long_run_variance_sets_omega() -> None:
    alpha = np.array([0.05, 0.1])
    garch = GarchProcess.from_long_run_variance(
        alpha=alpha, beta=0.85, long_run_variance=1e-4
    )

    np.testing.assert_allclose(garch.omega, 1e-4 * (1 - alpha - 0.85))
    np.testing.assert_allclose(garch.long_run_variance(), [1e-4, 1e-4])


def test_garch_from_long_run_variance_simulates_like_omega() -> None:
    from_omega = Simulator(vol=GARCH).simulate(50, first_date=FIRST_DATE, rng=9)
    from_variance = Simulator(
        vol=GarchProcess.from_long_run_variance(
            alpha=0.08, beta=0.9, long_run_variance=GARCH_LONG_RUN_VAR
        )
    ).simulate(50, first_date=FIRST_DATE, rng=9)

    np.testing.assert_allclose(
        from_variance.returns.values, from_omega.returns.values, rtol=1e-12
    )


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"alpha": 0.1, "beta": 0.8, "long_run_variance": 0.0}, "long-run variance"),
        ({"alpha": 0.1, "beta": 0.9, "long_run_variance": 1e-4}, "below 1"),
        ({"alpha": -0.1, "beta": 0.8, "long_run_variance": 1e-4}, "at least 0"),
    ],
)
def test_garch_from_long_run_variance_rejects_invalid_parameters(
    params: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        GarchProcess.from_long_run_variance(**params)


def test_ewma_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        EwmaProcess(decay=1.1, start_vol=0.01)
    with pytest.raises(ValueError, match="positive"):
        EwmaProcess(decay=0.94, start_vol=0.0)


def test_ar_rejects_unit_root() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ArProcess(ar=(0.4, 0.6))


def test_simulate_rejects_parameter_not_matching_batch_shape() -> None:
    vol = EwmaProcess(decay=np.array([0.9, 0.94, 0.97]), start_vol=0.01)

    with pytest.raises(ValueError, match=r"EwmaProcess.decay of shape \(3,\)"):
        Simulator(vol=vol).simulate(10, first_date=FIRST_DATE, batch_shape=(2,))


@pytest.mark.parametrize(("n_dates", "burn_in"), [(0, 0), (10, -1)])
def test_simulate_rejects_invalid_lengths(n_dates: int, burn_in: int) -> None:
    with pytest.raises(ValueError, match="at least"):
        Simulator(vol=GARCH).simulate(n_dates, first_date=FIRST_DATE, burn_in=burn_in)
