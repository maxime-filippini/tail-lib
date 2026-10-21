"""Kupiec's proportion-of-failures test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import special
from scipy import stats

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray
    from tail_lib._typing import IntArray
    from tail_lib.time_series import TimeSeries


@dataclass(frozen=True)
class KupiecResult:
    """The outcome of a Kupiec test, one value per series.

    Every array has the batch shape of the overshootings tested.
    """

    n_obs: IntArray
    """Number of dates with an overshooting flag (1.0 or 0.0)."""

    n_overshootings: IntArray
    """Number of overshootings."""

    expected_rate: float
    """Overshooting rate under the model, `1 - cf`."""

    observed_rate: FloatArray
    """Observed overshooting rate, `n_overshootings / n_obs`."""

    statistic: FloatArray
    """Likelihood-ratio statistic, chi-squared with 1 degree of freedom."""

    p_value: FloatArray
    """Probability of a statistic at least as large under the model."""


def kupiec_pof(overshootings: TimeSeries, cf: float) -> KupiecResult:
    """Test whether the overshooting rate matches the confidence level.

    Under the model, each date is an overshooting with probability `p = 1 - cf`.
    With `x` overshootings out of `n` dates, the likelihood ratio of the
    model's rate `p` against the observed rate `x / n` is

        LR = -2 * [ (n - x) * ln(1 - p) + x * ln(p)
                    - (n - x) * ln(1 - x / n) - x * ln(x / n) ]

    with `0 * ln(0) = 0`. Under the model, `LR` follows a chi-squared
    distribution with 1 degree of freedom. The test only looks at how many
    overshootings there are, not at when they happen.

    Reference: Kupiec, P. H. (1995). Techniques for verifying the accuracy of
    risk measurement models. The Journal of Derivatives, 3(2), 73-84.

    Args:
        overshootings (TimeSeries): The overshooting flags, see `overshootings`.
            `NaN` dates are left out.
        cf (float): The confidence level of the value-at-risk, in (0,1).

    Returns:
        KupiecResult: The test outcome, one value per series. A series with no
            flagged dates has `NaN` rates, statistic and p-value.

    Raises:
        ValueError: If `cf` is not in (0,1).

    """
    if cf <= 0 or cf >= 1:
        msg = "Confidence level has to be in (0,1)!"
        raise ValueError(msg)

    flags = overshootings.values
    n = np.sum(~np.isnan(flags), axis=0)
    x = np.nansum(flags, axis=0).astype(np.int64)
    p = 1 - cf

    with np.errstate(invalid="ignore", divide="ignore"):
        observed = x / n
        log_l_model = special.xlogy(n - x, 1 - p) + special.xlogy(x, p)
        log_l_observed = special.xlogy(n - x, 1 - observed) + special.xlogy(x, observed)
    statistic = -2 * (log_l_model - log_l_observed)

    return KupiecResult(
        n_obs=np.asarray(n, dtype=np.int64),
        n_overshootings=np.asarray(x),
        expected_rate=p,
        observed_rate=np.asarray(observed),
        statistic=np.asarray(statistic),
        p_value=np.asarray(stats.chi2.sf(statistic, df=1)),
    )
