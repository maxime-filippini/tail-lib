"""Standardized distributions: zero mean and unit variance.

A return with mean `mu` and volatility `sigma` is `mu + sigma * z`, with `z`
drawn from one of these. The simulator draws its shocks from them; parametric
value-at-risk and expected shortfall models will later read their quantiles.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tail_lib._checks import check_broadcasts

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray


class Distribution(ABC):
    """A distribution with zero mean and unit variance.

    Its parameters are single values or arrays broadcastable to a batch shape,
    e.g. one per series.
    """

    @abstractmethod
    def draw(self, rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
        """Draw independent values.

        Args:
            rng (np.random.Generator): The random generator.
            shape (tuple[int, ...]): The shape of the draws. The parameters
                broadcast against its last axes.

        Returns:
            FloatArray: The draws.

        """

    @abstractmethod
    def check_shape(self, batch_shape: tuple[int, ...]) -> None:
        """Check every parameter broadcasts to the batch shape.

        Args:
            batch_shape (tuple[int, ...]): The batch shape.

        Raises:
            ValueError: If a parameter doesn't broadcast to it.

        """


@dataclass(frozen=True)
class Gaussian(Distribution):
    """The standard normal distribution. It has no parameters."""

    def draw(self, rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:  # noqa: D102
        return rng.standard_normal(shape)

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:  # noqa: D102
        pass


@dataclass(frozen=True)
class StudentT(Distribution):
    """The Student-t distribution, scaled to unit variance.

    A Student-t value with `df` degrees of freedom has variance
    `df / (df - 2)`, so it is multiplied by `sqrt((df - 2) / df)`. The lower
    `df`, the fatter the tails.

    Reference: Bollerslev, "A conditionally heteroskedastic time series model
    for speculative prices and rates of return", Review of Economics and
    Statistics, 1987.

    Args:
        df (FloatArray | float): The degrees of freedom, above 2.

    """

    df: FloatArray | float

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.df) <= 2):  # noqa: PLR2004
            msg = "Student-t degrees of freedom have to be above 2!"
            raise ValueError(msg)

    def draw(self, rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:  # noqa: D102
        df = np.asarray(self.df)
        return rng.standard_t(df, size=shape) * np.sqrt((df - 2) / df)

    def check_shape(self, batch_shape: tuple[int, ...]) -> None:  # noqa: D102
        check_broadcasts("StudentT", "df", self.df, batch_shape)
