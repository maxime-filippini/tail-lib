"""The node graph: preprocessors, and the cache that computes them."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tail_lib.time_series import TimeSeries


class Preprocessor(ABC):
    """A transformation of a returns time series into a model input.

    Subclasses implement `evaluate`, which gets the inputs of the preprocessor
    from a `Cache`, then computes its output. The computation itself usually
    lives in a `compute` method, whose arguments are the inputs.
    """

    @abstractmethod
    def evaluate(self, cache: Cache) -> TimeSeries:
        """Get the inputs from the cache, then compute the output.

        Args:
            cache (Cache): Holds the returns, and the outputs of other
                preprocessors on them.

        Returns:
            TimeSeries: The output of this preprocessor.

        """

    def run(self, returns: TimeSeries) -> TimeSeries:
        """Compute this preprocessor on the returns, with a fresh cache.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            TimeSeries: The output of this preprocessor.

        """
        return Cache(returns).get(self)


class Cache:
    """Outputs of preprocessors on one returns time series.

    Each preprocessor is computed at most once per cache. Preprocessors are
    identified by instance: share one instance between models to share its
    computation.
    """

    def __init__(self, returns: TimeSeries) -> None:
        """Build an empty cache for a returns time series.

        Args:
            returns (TimeSeries): The returns every preprocessor is computed on.

        """
        self._returns = returns
        self._outputs: dict[Preprocessor, TimeSeries] = {}
        self._pending: set[Preprocessor] = set()

    @property
    def returns(self) -> TimeSeries:
        """The returns every preprocessor is computed on."""
        return self._returns

    def get(self, preprocessor: Preprocessor) -> TimeSeries:
        """Get the output of a preprocessor, computing it on first request.

        Args:
            preprocessor (Preprocessor): The preprocessor.

        Returns:
            TimeSeries: The output of the preprocessor.

        Raises:
            ValueError: If the preprocessor needs its own output as an input.

        """
        if preprocessor in self._outputs:
            return self._outputs[preprocessor]

        if preprocessor in self._pending:
            msg = f"{type(preprocessor).__name__} depends on itself!"
            raise ValueError(msg)

        self._pending.add(preprocessor)
        try:
            output = preprocessor.evaluate(self)
        finally:
            self._pending.discard(preprocessor)

        self._outputs[preprocessor] = output
        return output
