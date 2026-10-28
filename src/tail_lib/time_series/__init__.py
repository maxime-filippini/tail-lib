"""Time series and the windows over them."""

from .time_series import TimeSeries
from .windows import Last
from .windows import Trailing

__all__ = ["Last", "TimeSeries", "Trailing"]
