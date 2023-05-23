"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 3], [3, 4], [5, 5]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and  integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 2, 1], [2, 3, 3], [9, 5, 4]], [9, 5, 4]),
        ([[1, 2, -3], [3, 4, -9], [5, 6, -1]], [5, 6, -1]),
    ])
def test_daily_max(test, expected):
    """Test that max function works for an array of zeros and integers."""
    from inflammation.models import daily_max
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 2, 1], [2, 3, 3], [9, 5, 4]], [0, 2, 1]),
        ([[1, 2, -3], [3, 4, -9], [5, 6, -1]], [1, 2, -9]),
])
def test_daily_min(test, expected):
    """Test that min function works for an array of zeros and  integers."""
    from inflammation.models import daily_min
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


def test_daily_min_integers():
    """Test that min function works for an array of positive integers."""
    from inflammation.models import daily_min

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([1, 2])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
