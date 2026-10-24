"""Shape checks shared by classes with array parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray


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
