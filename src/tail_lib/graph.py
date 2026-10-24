"""The node graph: nodes, the cache that computes them, and pools of nodes.

A node is one computation on a returns time series: a preprocessor (a model
input such as a volatility), a value-at-risk model, or a backtest. A node gets
its inputs, the outputs of other nodes, from a `Cache`, which computes each
node at most once.

Data from outside a run, e.g. value-at-risk forecasts computed earlier, enters
it as named inputs (R13): the caller passes them to the cache by name, and an
input node reads its own name.
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

    Subclasses also implement `dependencies`, which lists the nodes `evaluate`
    reads, so a graph of nodes can be walked without computing it.
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

    @abstractmethod
    def dependencies(self) -> Dependencies:
        """List the nodes `evaluate` gets from the cache.

        Each is keyed by the constructor argument it comes from, e.g.
        `{"vol_model": ..., "mean_model": ...}` for a Gaussian model. The
        returns and the named run inputs are not nodes, so they are not listed.

        Returns:
            Dependencies: The nodes this node reads, by name.

        """

    def run(
        self,
        returns: TimeSeries,
        inputs: Mapping[str, TimeSeries] | None = None,
    ) -> T:
        """Compute this node on the returns, with a fresh cache.

        Args:
            returns (TimeSeries): The returns time series.
            inputs (Mapping[str, TimeSeries] | None): The named inputs of the
                run, each with the dates and shape of `returns`.

        Returns:
            T: The output of this node.

        """
        return Cache(returns, inputs).get(self)


# The nodes a node reads have different output types, which a mapping can't
# express one by one: `Any` stands for each node's own output type.
type Dependencies = Mapping[str, Node[Any]]
"""The nodes a node reads, by name, see `Node.dependencies`."""


class Preprocessor(Node[TimeSeries]):
    """A node turning the returns into a model input, as a time series."""


class Cache:
    """Outputs of nodes on one returns time series.

    Each node is computed at most once per cache. Nodes are identified by
    instance: share one instance between nodes to share its computation.

    The cache also holds the named inputs of the run. Inputs no node reads are
    ignored.
    """

    def __init__(
        self,
        returns: TimeSeries,
        inputs: Mapping[str, TimeSeries] | None = None,
    ) -> None:
        """Build an empty cache for a returns time series.

        Args:
            returns (TimeSeries): The returns every node is computed on.
            inputs (Mapping[str, TimeSeries] | None): The named inputs of the
                run, each with the dates and shape of `returns`.

        """
        self._returns = returns
        self._inputs = {} if inputs is None else dict(inputs)
        # Nodes have different output types: `get` restores the type of each.
        self._outputs: dict[Node[Any], Any] = {}
        self._pending: set[Node[Any]] = set()

    @property
    def returns(self) -> TimeSeries:
        """The returns every node is computed on."""
        return self._returns

    def input(self, name: str) -> TimeSeries:
        """Get a named input of the run.

        Args:
            name (str): The name of the input.

        Returns:
            TimeSeries: The input, with the dates and shape of the returns.

        Raises:
            KeyError: If no input has this name.
            ValueError: If the input does not match the returns.

        """
        if name not in self._inputs:
            msg = f"Missing input {name!r}! Inputs given: {sorted(self._inputs)}."
            raise KeyError(msg)

        value = self._inputs[name]
        try:
            self._returns.check_aligned(value)
        except ValueError as error:
            msg = f"Input {name!r} does not match the returns: {error}"
            raise ValueError(msg) from error
        return value

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

    def run(
        self,
        returns: TimeSeries,
        inputs: Mapping[str, TimeSeries] | None = None,
    ) -> dict[str, T]:
        """Compute every node of the pool on the returns, with a fresh cache.

        Args:
            returns (TimeSeries): The returns time series.
            inputs (Mapping[str, TimeSeries] | None): The named inputs of the
                run, each with the dates and shape of `returns`.

        Returns:
            dict[str, T]: The output of each node, keyed by name.

        """
        return self.evaluate(Cache(returns, inputs))
