import numpy as np
import pytest
from scipy import stats

from tail_lib.distributions import Distribution
from tail_lib.distributions import Gaussian
from tail_lib.distributions import StudentT


@pytest.mark.parametrize(
    "distribution", [Gaussian(), StudentT(df=np.array([4.0, 10.0]))]
)
def test_draws_have_zero_mean_and_unit_variance(distribution: Distribution) -> None:
    z = distribution.draw(np.random.default_rng(7), (200_000, 2))

    np.testing.assert_allclose(z.mean(axis=0), [0.0, 0.0], atol=0.01)
    np.testing.assert_allclose(z.var(axis=0), [1.0, 1.0], rtol=0.05)


def test_student_t_rejects_low_degrees_of_freedom() -> None:
    with pytest.raises(ValueError, match="above 2"):
        StudentT(df=2.0)


def test_student_t_rejects_df_not_matching_batch_shape() -> None:
    with pytest.raises(ValueError, match=r"StudentT.df of shape \(3,\)"):
        StudentT(df=np.array([4.0, 5.0, 6.0])).check_shape((2,))


def test_gaussian_quantile_is_standard_normal() -> None:
    np.testing.assert_allclose(Gaussian().quantile(0.01), stats.norm.ppf(0.01))


def test_student_t_quantile_is_scaled_to_unit_variance() -> None:
    df = np.array([4.0, 10.0])

    expected = stats.t.ppf(0.01, df) * np.sqrt((df - 2) / df)

    np.testing.assert_allclose(StudentT(df=df).quantile(0.01), expected)


@pytest.mark.parametrize(
    "distribution", [Gaussian(), StudentT(df=np.array([4.0, 10.0]))]
)
def test_quantile_matches_draws(distribution: Distribution) -> None:
    z = distribution.draw(np.random.default_rng(8), (400_000, 2))

    np.testing.assert_allclose(
        np.quantile(z, 0.01, axis=0), distribution.quantile(0.01), rtol=0.03
    )


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1])
def test_quantile_rejects_probability_outside_unit_interval(q: float) -> None:
    with pytest.raises(ValueError, match=r"in \(0,1\)"):
        Gaussian().quantile(q)
