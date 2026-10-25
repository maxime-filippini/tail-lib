"""Checks of parameter values, shared across the library, so each is written once."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray


def check_probability(name: str, value: float) -> None:
    """Check that a probability, e.g. a confidence level, is in (0,1).

    Args:
        name (str): What the value is, for the message, e.g. "Confidence level".
        value (float): The value.

    Raises:
        ValueError: If it isn't in (0,1).

    """
    if not 0 < value < 1:
        msg = f"{name} has to be in (0,1), got {value}!"
        raise ValueError(msg)


def check_name(kind: str, name: str) -> None:
    """Check that a name, e.g. a model's, is a Python identifier.

    Names appear in dotted paths through a run's graph, e.g.
    `models.gauss.vol_model`, so a name can't hold a dot, a space or a dash.

    Args:
        kind (str): What the name is, for the message, e.g. "Model".
        name (str): The name.

    Raises:
        ValueError: If it isn't a Python identifier.

    """
    if not name.isidentifier():
        msg = (
            f"{kind} name has to be a Python identifier (letters, digits and "
            f"underscores, not starting with a digit), got {name!r}!"
        )
        raise ValueError(msg)


def check_unit_interval(name: str, value: FloatArray | float) -> None:
    """Check that a value, e.g. a decay factor, is in [0,1], every element of it.

    Args:
        name (str): What the value is, for the message, e.g. "Decay factor".
        value (FloatArray | float): The value, a single value or an array.

    Raises:
        ValueError: If an element isn't in [0,1].

    """
    array = np.asarray(value)
    if np.any(array < 0) or np.any(array > 1):
        msg = f"{name} has to be in [0,1]!"
        raise ValueError(msg)


def check_positive(name: str, value: FloatArray | float) -> None:
    """Check that a value, e.g. a volatility, is positive, every element of it.

    Args:
        name (str): What the value is, for the message, e.g. "Starting
            volatility".
        value (FloatArray | float): The value, a single value or an array.

    Raises:
        ValueError: If an element isn't positive.

    """
    if np.any(np.asarray(value) <= 0):
        msg = f"{name} has to be positive!"
        raise ValueError(msg)


def check_broadcasts(
    owner: str, name: str, value: FloatArray | float, batch_shape: tuple[int, ...]
) -> None:
    """Check that a parameter broadcasts to a batch shape, without growing it.

    Args:
        owner (str): The class the parameter belongs to, for the message.
        name (str): The name of the parameter, for the message.
        value (FloatArray | float): The parameter.
        batch_shape (tuple[int, ...]): The batch shape it has to broadcast to.

    Raises:
        ValueError: If it doesn't.

    """
    shape = np.shape(value)
    try:
        broadcast = np.broadcast_shapes(shape, batch_shape)
    except ValueError:
        broadcast = None
    if broadcast != batch_shape:
        msg = (
            f"{owner}.{name} of shape {shape} does not broadcast to the "
            f"batch shape {batch_shape}!"
        )
        raise ValueError(msg)
