import datetime

import numpy as np

from tail_lib.calendar import Calendar
from tail_lib.time_series.time_series import TimeSeries

# 2020-01-03 is a Friday.
FRIDAY = datetime.date(2020, 1, 3)
MONDAY = datetime.date(2020, 1, 6)


def _dates(*days: str) -> np.ndarray:
    return np.array([f"2020-01-{d}" for d in days], dtype="datetime64[D]")


def test_from_first_date_uses_weekdays_by_default() -> None:
    ts = TimeSeries.from_first_date(FRIDAY, values=np.zeros(3))

    np.testing.assert_array_equal(ts.dates, _dates("03", "06", "07"))


def test_to_last_date_uses_weekdays_by_default() -> None:
    ts = TimeSeries.to_last_date(MONDAY, values=np.zeros(3))

    np.testing.assert_array_equal(ts.dates, _dates("02", "03", "06"))


def test_date_helpers_use_given_calendar() -> None:
    calendar = Calendar(weekmask="1111111")

    np.testing.assert_array_equal(
        TimeSeries.from_first_date(FRIDAY, np.zeros(3), calendar=calendar).dates,
        _dates("03", "04", "05"),
    )
    np.testing.assert_array_equal(
        TimeSeries.to_last_date(MONDAY, np.zeros(3), calendar=calendar).dates,
        _dates("04", "05", "06"),
    )


def test_date_helpers_use_first_axis_of_values() -> None:
    ts = TimeSeries.from_first_date(FRIDAY, values=np.zeros((4, 2)))

    assert ts.dates.shape == (4,)
