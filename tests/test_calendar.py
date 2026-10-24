import datetime

import numpy as np
import pytest

from tail_lib.calendar import Calendar

# 2020-01-03 is a Friday.
FRIDAY = datetime.date(2020, 1, 3)
SATURDAY = datetime.date(2020, 1, 4)
SUNDAY = datetime.date(2020, 1, 5)
MONDAY = datetime.date(2020, 1, 6)


def _dates(*days: str) -> np.ndarray:
    return np.array([f"2020-01-{d}" for d in days], dtype="datetime64[D]")


def test_default_calendar_skips_weekends() -> None:
    dates = Calendar().dates_from(FRIDAY, 3)

    assert dates.dtype == np.dtype("datetime64[D]")
    np.testing.assert_array_equal(dates, _dates("03", "06", "07"))


def test_dates_to_ends_on_end_date() -> None:
    np.testing.assert_array_equal(
        Calendar().dates_to(MONDAY, 3), _dates("02", "03", "06")
    )


def test_dates_from_invalid_start_rolls_forward() -> None:
    np.testing.assert_array_equal(
        Calendar().dates_from(SATURDAY, 2), _dates("06", "07")
    )


def test_dates_to_invalid_end_rolls_backward() -> None:
    np.testing.assert_array_equal(Calendar().dates_to(SUNDAY, 2), _dates("02", "03"))


def test_holidays_are_skipped() -> None:
    calendar = Calendar(holidays=[MONDAY])

    np.testing.assert_array_equal(
        calendar.dates_from(FRIDAY, 3), _dates("03", "07", "08")
    )
    np.testing.assert_array_equal(
        calendar.dates_to(datetime.date(2020, 1, 7), 3), _dates("02", "03", "07")
    )


def test_all_days_weekmask_gives_consecutive_days() -> None:
    calendar = Calendar(weekmask="1111111")

    np.testing.assert_array_equal(
        calendar.dates_from(FRIDAY, 4), _dates("03", "04", "05", "06")
    )


def test_accepts_numpy_dates() -> None:
    np.testing.assert_array_equal(
        Calendar().dates_from(np.datetime64(FRIDAY), 2),
        Calendar().dates_from(FRIDAY, 2),
    )


def test_zero_dates_gives_empty_array() -> None:
    assert Calendar().dates_from(FRIDAY, 0).shape == (0,)
    assert Calendar().dates_to(FRIDAY, 0).shape == (0,)


def test_negative_number_of_dates_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Calendar().dates_from(FRIDAY, -1)
