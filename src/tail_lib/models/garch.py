"""The GARCH(1, 1) method: its variance recursion.

Only the recursion for now, shared by the simulator (`tail_lib.simulation`)
and a future GARCH volatility model, so the formula is written once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tail_lib._typing import FloatArray


def garch_update(
    var: FloatArray,
    eps: FloatArray,
    *,
    omega: FloatArray | float,
    alpha: FloatArray | float,
    beta: FloatArray | float,
) -> FloatArray:
    """One step of the GARCH(1, 1) variance recursion.

    Updates the variance forecast with the residual of one date:

        omega + alpha * eps ** 2 + beta * var

    Reference: Bollerslev, "Generalized autoregressive conditional
    heteroskedasticity", Journal of Econometrics, 1986.

    Args:
        var (FloatArray): The variance forecast for the date of `eps`.
        eps (FloatArray): The residual of one date: its return minus its mean.
        omega (FloatArray | float): The constant of the recursion.
        alpha (FloatArray | float): The weight of the squared residual.
        beta (FloatArray | float): The weight of the previous variance.

    Returns:
        FloatArray: The variance forecast for the next date.

    """
    return omega + alpha * eps**2 + beta * var
