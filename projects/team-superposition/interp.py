import numpy as np

def interpolate_nan_intervals(arr):
    """
    Replace intervals of NaN values in a numpy array with linearly interpolated values
    from the first and last non-NaN values surrounding the interval using np.interp.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Input array that may contain NaN values
        
    Returns:
    --------
    numpy.ndarray
        Array with NaN intervals replaced by linearly interpolated values
        
    Raises:
    -------
    ValueError
        If the first or last element of the array is NaN
    """
    # Make a copy to avoid modifying the original
    result = np.copy(arr)
    
    # Check if first or last element is NaN
    if np.isnan(result[0]) or np.isnan(result[-1]):
        raise ValueError("The first or last element of the array is NaN, cannot interpolate.")
    
    # Get indices of NaN and non-NaN values
    nan_mask = np.isnan(result)
    if not np.any(nan_mask):
        return result  # No NaNs to interpolate
    
    # Get indices of all elements (for x-coordinates)
    indices = np.arange(len(result))
    
    # Get indices and values of non-NaN elements (for reference points)
    valid_indices = indices[~nan_mask]
    valid_values = result[~nan_mask]
    
    # Interpolate NaN values based on their indices
    result[nan_mask] = np.interp(indices[nan_mask], valid_indices, valid_values)
    
    return result