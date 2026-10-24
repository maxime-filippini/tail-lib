"""Calendars, deciding which dates are valid when filling in dates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import datetime
    from collections.abc import Iterable

    from tail_lib._typing import DateArray1D


class Calendar:
    """A set of valid dates, defined by valid weekdays and holidays.

    A date is valid if its weekday is in `weekmask` and it is not in
    `holidays`. The default is every Monday to Friday, without holidays.

    This wraps `numpy.busdaycalendar`, and generates dates with
    `numpy.busday_offset`.
    """

    def __init__(
        self,
        weekmask: str = "Mon Tue Wed Thu Fri",
        holidays: Iterable[datetime.date | np.datetime64] = (),
    ) -> None:
        """Build the calendar.

        Args:
            weekmask (str): The valid weekdays, in any format accepted by
                `numpy.busdaycalendar`: e.g. "Mon Tue Wed Thu Fri" or "1111100".
            holidays (Iterable[datetime.date | np.datetime64]): Dates that are
                not valid, whatever their weekday.

        """
        self._busdaycal = np.busdaycalendar(
            weekmask=weekmask,
            holidays=np.array(list(holidays), dtype="datetime64[D]"),
        )

    @property
    def weekmask(self) -> np.ndarray:
        """The valid weekdays, as 7 booleans from Monday to Sunday."""
        return self._busdaycal.weekmask

    @property
    def holidays(self) -> DateArray1D:
        """The holidays, sorted, keeping only those on valid weekdays."""
        return self._busdaycal.holidays

    def dates_from(
        self,
        start: datetime.date | np.datetime64,
        n: int,
    ) -> DateArray1D:
        """Build `n` consecutive valid dates, starting from `start`.

        If `start` is not valid, the dates start from the next valid date.

        Args:
            start (datetime.date | np.datetime64): The first date.
            n (int): The number of dates.

        Returns:
            DateArray1D: The dates, increasing.

        """
        _check_n_dates(n)
        return np.busday_offset(
            np.datetime64(start, "D"),
            np.arange(n),
            roll="forward",
            busdaycal=self._busdaycal,
        )

    def dates_to(
        self,
        end: datetime.date | np.datetime64,
        n: int,
    ) -> DateArray1D:
        """Build `n` consecutive valid dates, ending on `end`.

        If `end` is not valid, the dates end on the previous valid date.

        Args:
            end (datetime.date | np.datetime64): The last date.
            n (int): The number of dates.

        Returns:
            DateArray1D: The dates, increasing.

        """
        _check_n_dates(n)
        return np.busday_offset(
            np.datetime64(end, "D"),
            np.arange(1 - n, 1),
            roll="backward",
            busdaycal=self._busdaycal,
        )


def _check_n_dates(n: int) -> None:
    if n < 0:
        msg = f"Number of dates has to be non-negative, got {n}!"
        raise ValueError(msg)
