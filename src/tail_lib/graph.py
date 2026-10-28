"""The node graph: value-identified nodes and their declared inputs.

A node is one computation on a returns time series: a preprocessor (a model
input such as a volatility), a value-at-risk model, or a backtest. A `Context`
computes each node at most once: first its inputs, the nodes in its fields,
then the node's `compute` on their outputs.

Nodes are frozen dataclasses, identified by value: two nodes of the same class
with equal constructor arguments, their input nodes included, are one node
(D9). A node's fields holding nodes are its dependencies, the others its
settings.

Data from outside a run, e.g. value-at-risk forecasts computed earlier, enters
it as named inputs (R13): the caller passes them to the context by name, and
it gives each input node (a `RunInput`) the input of its name as its output,
without computing it.
"""

from __future__ import annotations

import inspect
from abc import ABC
from abc import abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from functools import cache
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from tail_lib._checks import check_name_is_identifier
from tail_lib.time_series.time_series import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tail_lib._typing import FloatArray
    from tail_lib._typing import Floats
    from tail_lib.context import Context
    from tail_lib.models.ewma import EwmaStart
    from tail_lib.time_series.windows import Last
    from tail_lib.time_series.windows import Trailing


# The context of the innermost open `with Context():` block, which nodes built
# in it bind to (D10), see `tail_lib.context`. A `ContextVar`, not a global:
# each thread and asyncio task sees its own, so concurrent requests in a
# service (R17) never mix their nodes. It lives here, not in `context.py`,
# because `Node` reads it and `context.py` imports `Node`.
current_context: ContextVar[Context | None] = ContextVar(
    "tail_lib_context", default=None
)


@dataclass(frozen=True)
class Node[T](ABC):
    """A computation on a returns time series, with an output of type `T`.

    Subclasses are frozen dataclasses, so a node is identified by value: its
    class and its fields, which are its constructor arguments (D9). A field
    excluded from comparison (`compare=False`) is not part of it, e.g. the
    node's `name`.

    The fields holding nodes are the node's `dependencies`, so a graph of
    nodes can be walked without computing it. The other constructor
    arguments that are part of its identity are its `settings`, so the graph
    can be described, e.g. in a report (R19). Both are read from the fields.

    Any node can be given a `name`, a Python identifier (R20), to be found
    by it, e.g. a value-at-risk model in a context. The name is not part of
    the node's identity: two equal nodes with different names are one node,
    computed once. A run input's name is the input it reads, and is part of
    its identity, see `RunInput`.

    Subclasses implement `compute`, which is pure (see `CLAUDE.md`): it takes
    the returns, then one keyword argument per dependency, named after its
    field, holding that dependency's output, and uses nothing else but the
    node's own settings. E.g. a Gaussian model, with fields `cf`, `vol` and
    `mean`, has `compute(self, returns, /, *, vol, mean)`. A `Context`
    computes the dependencies first, then calls `compute` with their outputs.

    Building a node checks that `compute` takes exactly its dependencies, by
    name (D11), so a mismatch fails where the model is declared, not
    mid-run. Subclasses with a `__post_init__` call `super().__post_init__()`
    for this. Run inputs (`RunInput`) are read from the run, never computed.

    A node built inside a `with Context():` block binds to that context, once
    its checks pass (D10). One built outside any context belongs to none, and
    can still be the input of a node in any context.

    Attributes:
        name (str | None): The node's name, keyword-only. `None` by default.

    """

    # Not part of the identity, nor of the repr, which shows the identity.
    name: str | None = field(default=None, kw_only=True, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is not None:
            check_name_is_identifier("Node", self.name)
        self._check_compute()
        context = current_context.get()
        if context is not None:
            context.bind(self)

    # Each node takes its own inputs, so `compute` has no common signature.
    # The typing spec reads `*args: Any, **kwargs: Any` like `...`: every
    # subclass can narrow it, and ty checks direct calls to the subclass's.
    # Only the call in `Context.run` is unchecked, hence `_check_compute`.
    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
        """Compute the output from the returns and the outputs of the inputs.

        Args:
            *args (Any): The returns, a `TimeSeries`.
            **kwargs (Any): The output of each dependency, by field name.

        Returns:
            T: The output of this node.

        """

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
        """List the nodes whose outputs `compute` takes: the fields holding nodes.

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
        argument that is part of the node's identity: not its `name`.

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

        Fields the node derives itself (`init=False`), and fields that aren't
        part of its identity (`compare=False`), e.g. its `name`, are left out.
        """
        return {
            node_field.name: getattr(self, node_field.name)
            for node_field in fields(self)
            if node_field.init and node_field.compare
        }


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
    The context gives it the run input of its name as its output, and never
    calls its `compute`.

    Its name is both the node's name and the input it reads, e.g.
    `ValueAtRiskInput("var_ext", cf=0.99)` is the model `var_ext`, reading
    the run input `var_ext`. Unlike other nodes' names, it is part of the
    node's identity, and a setting: inputs of different names are different
    nodes.

    Building one raises `ValueError` if the name isn't a Python identifier.

    Attributes:
        name (str): The name of the input, as passed to the run: a Python
            identifier.

    """

    # Replaces `Node.name`: required, positional, and part of the identity.
    # `field()` keeps it required: a bare annotation would inherit the
    # `None` default `Node` sets on the class. Subclasses list `RunInput`
    # first among their bases, so this field wins over `Node.name` again.
    name: str = field()

    def __post_init__(self) -> None:
        check_name_is_identifier("Input", self.name)
        super().__post_init__()

    # Every `compute` takes the returns first, even when it doesn't use them.
    def compute(self, returns: TimeSeries, /) -> TimeSeries:  # noqa: ARG002
        """Refuse to compute: the output is the run input, see `Context.run`.

        Raises:
            TypeError: Always.

        """
        msg = f"Input {self.name!r} is given to the run, not computed!"
        raise TypeError(msg)
