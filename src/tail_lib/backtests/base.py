"""The parent class of backtest declarations, e.g. `Kupiec`."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from tail_lib._checks import check_name_is_identifier
from tail_lib.graph import current_context

if TYPE_CHECKING:
    from tail_lib.graph import Backtest
    from tail_lib.models.value_at_risk import ValueAtRiskModel


@dataclass(frozen=True, kw_only=True)
class BacktestDeclaration(ABC):
    """Declare a test to apply to every model of a context.

    A declaration only defines the settings of the backtest test. It is not
    a node. The reason is that the node would have an explicit dependency on
    the overshootings associated with a model, which would force the user
    to define these overshooting nodes.

    Instead, when the graph associated with the context is built, these nodes
    are created dynamically.

    Subclasses add their settings as fields, e.g. a window, and implement
    `_bind`. A subclass with its own `__post_init__` calls this one.

    Attributes:
        name (str | None): Registration name, a Python identifier.
            Required inside a context, optional outside. Not part of the
            declaration's value identity.

    """

    name: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is not None:
            check_name_is_identifier("Backtest", self.name)

        context = current_context.get()
        if context is not None:
            if self.name is None:
                kind = type(self).__name__
                msg = (
                    f"{kind} is built in a context, so it needs a name, "
                    f'e.g. `name="{kind.lower()}"`!'
                )
                raise ValueError(msg)
            context.add_backtest(self.name, self)

    @abstractmethod
    def _bind(self, model: ValueAtRiskModel) -> Backtest[Any]:
        """Build the test node for one model, inside the context's walk."""
