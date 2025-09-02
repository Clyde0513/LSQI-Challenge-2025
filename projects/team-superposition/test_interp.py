
import numpy as np
from interp import interpolate_nans

def test_interpolate_nans_1d_middle_nan():
	arr = np.array([1.0, np.nan, 3.0])
	result = interpolate_nans(arr)
	expected = np.array([1.0, 2.0, 3.0])
	np.testing.assert_array_almost_equal(result, expected)

def test_interpolate_nans_1d_boundary_nan():
	arr = np.array([np.nan, 2.0, 3.0, np.nan])
	result = interpolate_nans(arr)
	expected = np.array([2.0, 2.0, 3.0, 3.0])
	np.testing.assert_array_almost_equal(result, expected)

def test_interpolate_nans_1d_all_nan():
	arr = np.array([np.nan, np.nan, np.nan])
	result = interpolate_nans(arr)
	expected = np.array([np.nan, np.nan, np.nan])
	np.testing.assert_array_equal(result, expected)

def test_interpolate_nans_1d_single_valid():
	arr = np.array([np.nan, 5.0, np.nan])
	result = interpolate_nans(arr)
	expected = np.array([5.0, 5.0, 5.0])
	np.testing.assert_array_equal(result, expected)

def test_interpolate_nans_1d_no_nan():
	arr = np.array([1.0, 2.0, 3.0])
	result = interpolate_nans(arr)
	np.testing.assert_array_equal(result, arr)

def test_interpolate_nans_2d_axis0():
	arr = np.array([[1.0, np.nan], [3.0, 4.0]])
	result = interpolate_nans(arr, axis=0)
	expected = np.array([[1.0, 4.0], [3.0, 4.0]])
	np.testing.assert_array_almost_equal(result, expected)

def test_interpolate_nans_2d_axis1():
	arr = np.array([[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]])
	result = interpolate_nans(arr, axis=1)
	expected = np.array([[2.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	np.testing.assert_array_almost_equal(result, expected)

def test_interpolate_nans_no_boundary_correction():
	arr = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
	result = interpolate_nans(arr, correct_boundaries=False)
	expected = np.array([np.nan, 2.0, 3.0, 4.0, np.nan])
	# Compare non-NaN values and check NaN positions
	np.testing.assert_array_almost_equal(result[1:4], expected[1:4])
	assert np.isnan(result[0]) and np.isnan(result[4])

