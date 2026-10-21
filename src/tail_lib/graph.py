"""The node graph: nodes, the cache that computes them, and pools of nodes.

A node is one computation on a returns time series: a preprocessor (a model
input such as a volatility), a value-at-risk model, or a backtest. A node gets
its inputs, the outputs of other nodes, from a `Cache`, which computes each
node at most once.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from tail_lib.time_series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Mapping


class Node[T](ABC):
    """A computation on a returns time series, with an output of type `T`.

    Subclasses implement `evaluate`, which gets the inputs of the node from a
    `Cache`, then computes its output. The computation itself usually lives in
    a `compute` method, whose arguments are the inputs.
    """

    @abstractmethod
    def evaluate(self, cache: Cache) -> T:
        """Get the inputs from the cache, then compute the output.

        Args:
            cache (Cache): Holds the returns, and the outputs of other nodes
                on them.

        Returns:
            T: The output of this node.

        """

    def run(self, returns: TimeSeries) -> T:
        """Compute this node on the returns, with a fresh cache.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            T: The output of this node.

        """
        return Cache(returns).get(self)


class Preprocessor(Node[TimeSeries]):
    """A node turning the returns into a model input, as a time series."""


class Cache:
    """Outputs of nodes on one returns time series.

    Each node is computed at most once per cache. Nodes are identified by
    instance: share one instance between nodes to share its computation.
    """

    def __init__(self, returns: TimeSeries) -> None:
        """Build an empty cache for a returns time series.

        Args:
            returns (TimeSeries): The returns every node is computed on.

        """
        self._returns = returns
        # Nodes have different output types: `get` restores the type of each.
        self._outputs: dict[Node[Any], Any] = {}
        self._pending: set[Node[Any]] = set()

    @property
    def returns(self) -> TimeSeries:
        """The returns every node is computed on."""
        return self._returns

    def get[T](self, node: Node[T]) -> T:
        """Get the output of a node, computing it on first request.

        Args:
            node (Node[T]): The node.

        Returns:
            T: The output of the node.

        Raises:
            ValueError: If the node needs its own output as an input.

        """
        if node in self._outputs:
            return self._outputs[node]

        if node in self._pending:
            msg = f"{type(node).__name__} depends on itself!"
            raise ValueError(msg)

        self._pending.add(node)
        try:
            output = node.evaluate(self)
        finally:
            self._pending.discard(node)

        self._outputs[node] = output
        return output


class Pool[T]:
    """Named nodes computed on the same returns.

    The nodes share one cache, so a node they depend on in common is computed
    once across the pool. Pools evaluated on the same cache share it too.
    """

    def __init__(self, nodes: Mapping[str, Node[T]]) -> None:
        self._nodes = dict(nodes)

    def evaluate(self, cache: Cache) -> dict[str, T]:
        """Compute every node of the pool on a cache.

        Args:
            cache (Cache): The cache, possibly shared with other pools.

        Returns:
            dict[str, T]: The output of each node, keyed by name.

        """
        return {name: cache.get(node) for name, node in self._nodes.items()}

    def run(self, returns: TimeSeries) -> dict[str, T]:
        """Compute every node of the pool on the returns, with a fresh cache.

        Args:
            returns (TimeSeries): The returns time series.

        Returns:
            dict[str, T]: The output of each node, keyed by name.

        """
        return self.evaluate(Cache(returns))
