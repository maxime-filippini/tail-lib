"""The node graph: nodes, the cache that computes them, and pools of nodes.

A node is one computation on a returns time series: a preprocessor (a model
input such as a volatility), a value-at-risk model, or a backtest. A node gets
its inputs, the outputs of other nodes, from a `Cache`, which computes each
node at most once.

Nodes are frozen dataclasses, identified by value: two nodes of the same class
with equal constructor arguments, their input nodes included, are one node
(D9). A node's fields holding nodes are its dependencies, the others its
settings.

Data from outside a run, e.g. value-at-risk forecasts computed earlier, enters
it as named inputs (R13): the caller passes them to the cache by name, and the
cache gives each input node (a `RunInput`) the input of its name as its
output, without computing it.
"""

from __future__ import annotations

import inspect
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import fields
from functools import cache
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from tail_lib._checks import check_name
from tail_lib.time_series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tail_lib._typing import FloatArray
    from tail_lib._typing import Floats
    from tail_lib.models.ewma import EwmaStart
    from tail_lib.windows import Last
    from tail_lib.windows import Trailing


@dataclass(frozen=True)
class Node[T](ABC):
    """A computation on a returns time series, with an output of type `T`.

    Subclasses are frozen dataclasses, so a node is identified by value: its
    class and its fields, which are its constructor arguments (D9). A field
    excluded from comparison (`compare=False`) is not part of it.

    The fields holding nodes are the node's `dependencies`, so a graph of
    nodes can be walked without computing it. The other constructor
    arguments are its `settings`, so the graph can be described, e.g. in a
    report (R19). Both are read from the fields.

    Subclasses implement `compute`, which is pure (see `CLAUDE.md`): it takes
    the returns, then one keyword argument per dependency, named after its
    field, holding that dependency's output, and uses nothing else but the
    node's own settings. E.g. a Gaussian model, with fields `cf`, `vol` and
    `mean`, has `compute(self, returns, /, *, vol, mean)`. `evaluate` gets
    those outputs from a `Cache` and calls `compute`.

    Building a node checks that `compute` takes exactly its dependencies, by
    name (D11), so a mismatch fails where the model is declared, not
    mid-run. Subclasses with a `__post_init__` call `super().__post_init__()`
    for this. Run inputs (`RunInput`) are read from the run, never computed.
    """

    def __post_init__(self) -> None:
        if type(self).evaluate is Node.evaluate:
            self._check_compute()

    # Each node takes its own inputs, so `compute` has no common signature.
    # The typing spec reads `*args: Any, **kwargs: Any` like `...`: every
    # subclass can narrow it, and ty checks direct calls to the subclass's.
    # Only the call in `evaluate` is unchecked, hence `_check_compute`.
    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
        """Compute the output from the returns and the outputs of the inputs.

        Args:
            *args (Any): The returns, a `TimeSeries`.
            **kwargs (Any): The output of each dependency, by field name.

        Returns:
            T: The output of this node.

        """

    def evaluate(self, cache: Cache) -> T:
        """Get the output of each dependency from the cache, then compute.

        Args:
            cache (Cache): Holds the returns, and the outputs of other nodes
                on them.

        Returns:
            T: The output of this node.

        """
        inputs = {name: cache.get(node) for name, node in self.dependencies().items()}
        return self.compute(cache.returns, **inputs)

    def _check_compute(self) -> None:
        """Check that `compute` takes the returns, then exactly the dependencies.

        Raises:
            TypeError: If `compute` takes `*args` or `**kwargs`, not one
                positional argument for the returns, or keyword-only
                arguments other than the dependencies' names.

        """
        node_type = type(self).__name__
        signature = _compute_signature(type(self))
        if signature is None:
            msg = (
                f"{node_type}.compute has to take the returns, then its inputs "
                f"as keyword-only arguments, without *args or **kwargs!"
            )
            raise TypeError(msg)
        inputs = set(self.dependencies())
        if signature != inputs:
            msg = (
                f"{node_type}.compute has to take one keyword-only argument per "
                f"input field. Missing: {sorted(inputs - signature)}. Not input "
                f"fields: {sorted(signature - inputs)}."
            )
            raise TypeError(msg)

    def dependencies(self) -> Dependencies:
        """List the nodes `evaluate` gets from the cache: the fields holding nodes.

        Each is keyed by its field, e.g. `{"vol": ..., "mean": ...}` for a
        Gaussian model. The returns and the named run inputs are
        not nodes, so they are not listed.

        Returns:
            Dependencies: The nodes this node reads, by name.

        """
        return {
            name: value
            for name, value in self._constructor_arguments().items()
            if isinstance(value, Node)
        }

    def settings(self) -> Settings:
        """List the constructor arguments that are not nodes.

        Each is keyed by its field, with the value the node uses, defaults
        included, e.g. `{"cf": 0.99, "window": Trailing(250), "interpolation":
        "linear"}` for a historical simulation. Arrays are stored as nested
        tuples, see `frozen_floats`. Arguments that are nodes are in
        `dependencies` instead, so together they cover every constructor
        argument.

        Returns:
            Settings: The settings of this node, by name.

        """
        return {
            name: value
            for name, value in self._constructor_arguments().items()
            if not isinstance(value, Node)
        }

    # The fields hold settings and nodes of every type: `Any` stands for each.
    def _constructor_arguments(self) -> dict[str, Any]:
        """Give the fields set by the constructor, by name, with their values.

        Fields the node derives itself (`init=False`) are left out.
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.init
        }

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


# Node types are hashable, and their signatures don't change: read each once.
@cache
def _compute_signature(node_type: type[Node[Any]]) -> frozenset[str] | None:
    """Give the keyword-only arguments of a node type's `compute`.

    `None` if `compute` takes `*args` or `**kwargs`, or not exactly one
    positional argument (after `self`), the returns.
    """
    parameters = list(inspect.signature(node_type.compute).parameters.values())[1:]
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if any(p.kind in variadic for p in parameters):
        return None
    if sum(p.kind in positional for p in parameters) != 1:
        return None
    return frozenset(
        p.name for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY
    )


# The nodes a node reads have different output types, which a mapping can't
# express one by one: `Any` stands for each node's own output type.
type Dependencies = Mapping[str, Node[Any]]
"""The nodes a node reads, by name, see `Node.dependencies`."""

type Setting = float | str | Floats | Trailing | Last | EwmaStart | None
"""A constructor argument of a node that is not a node, see `Node.settings`.

The kinds of settings the library's nodes take: numbers (a confidence level, a
number of iterations), names, windows, EWMA starts, and tuples (bounds, or an
array of values per series, see `frozen_floats`). `None` is an unset option,
e.g. a backtest on the whole sample.
"""

type Settings = Mapping[str, Setting]
"""The settings of a node, by name, see `Node.settings`."""


def frozen_floats(values: FloatArray | Floats | float) -> Floats | float:
    """Store values given to a node in a form that can be hashed.

    Arrays can't be hashed, so a node identified by value (D9) stores them as
    nested tuples, one level per axis, which compare equal when the values
    are. A single value stays a float. `np.asarray` gives the array back.

    Args:
        values (FloatArray | Floats | float): A single value, or an array.

    Returns:
        Floats | float: The float, or the values as nested tuples.

    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return _nested_tuples(array.tolist())


# `tolist` gives nested lists of floats, which NumPy's stubs type as `Any`.
def _nested_tuples(values: list[Any]) -> Floats:
    return tuple(
        _nested_tuples(value) if isinstance(value, list) else float(value)
        for value in values
    )


class Preprocessor(Node[TimeSeries]):
    """A node turning the returns into a model input, as a time series."""


class Backtest[T](Node[T]):
    """A node testing a value-at-risk model, e.g. on its overshootings.

    Its output is a test result of type `T`, with one value per window, not a
    time series.
    """


@dataclass(frozen=True)
class RunInput(Node[TimeSeries]):
    """A node whose output is given to the run, by name, not computed (R13).

    E.g. volatilities or value-at-risk forecasts computed outside the library.
    The cache gives it the run input of its name as its output, see
    `Cache.input`, and never calls its `compute`.

    Building one raises `ValueError` if the name isn't a Python identifier.

    Attributes:
        name (str): The name of the input, as passed to the run: a Python
            identifier.

    """

    name: str

    def __post_init__(self) -> None:
        check_name("Input", self.name)
        super().__post_init__()

    # Every `compute` takes the returns first, even when it doesn't use them.
    def compute(self, returns: TimeSeries, /) -> TimeSeries:  # noqa: ARG002
        """Refuse to compute: the output is the run input, see `Cache.get`.

        Raises:
            TypeError: Always.

        """
        msg = f"Input {self.name!r} is given to the run, not computed!"
        raise TypeError(msg)


class Cache:
    """Outputs of nodes on one returns time series.

    Each node is computed at most once per cache. Nodes are identified by
    value (D9), so equal nodes, even built separately, share one computation.

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

    @property
    def input_names(self) -> frozenset[str]:
        """The names of the run inputs given to the cache."""
        return frozenset(self._inputs)

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
            KeyError: If the node is a run input that wasn't given.
            ValueError: If the node needs its own output as an input, or is
                a run input that does not match the returns.

        """
        if node in self._outputs:
            return self._outputs[node]

        # A run input's output is the input of its name, never computed (R13).
        if isinstance(node, RunInput):
            self._outputs[node] = self.input(node.name)
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
