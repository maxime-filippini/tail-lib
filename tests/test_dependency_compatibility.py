"""Smoke tests run against each supported NumPy dependency range."""

import numpy as np

import tail_lib


def test_tail_lib_imports_with_installed_numpy() -> None:
    """The package and a basic NumPy operation work in every tox environment."""
    assert tail_lib is not None
    np.testing.assert_array_equal(np.arange(3) + 1, np.array([1, 2, 3]))
