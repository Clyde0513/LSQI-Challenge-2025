import numpy as np
from numpy.typing import NDArray

def interpolate_nans(arr: NDArray, axis: int = -1, correct_boundaries: bool = True) -> NDArray:
    """
    Interpolation of NaNs along a specified axis.
    
    Args:
        arr (NDArray): Input array with NaNs.
        axis (int): Axis along which to interpolate.
        correct_boundaries (bool): If True, propagate edge values to NaNs.
        
    Returns:
        NDArray: Array with NaNs interpolated.
    """
    arr = np.asarray(arr, dtype=float)
    result = arr.copy()
    
    if result.ndim == 1:
        return _interpolate_1d(result, correct_boundaries)
    
    # Move target axis to last axis for simpler broadcasting
    result = np.moveaxis(result, axis, -1)
    shape = result.shape
    N = shape[-1]
    
    if not np.any(np.isnan(result)):
        return np.moveaxis(result, -1, axis)
    
    idx = np.arange(N)
    nan_mask = np.isnan(result)
    
    if correct_boundaries:
        valid_mask = ~nan_mask
        has_valid = np.any(valid_mask, axis=-1)
        
        if np.any(has_valid):
            first_valid = np.full(has_valid.shape, -1, dtype=int)
            last_valid = np.full(has_valid.shape, -1, dtype=int)

            first_valid[has_valid] = np.argmax(valid_mask[has_valid], axis=-1)
            last_valid[has_valid] = N - 1 - np.argmax(np.flip(valid_mask[has_valid], axis=-1), axis=-1)
            
            it = np.ndindex(has_valid.shape)
            for ind in it:
                if has_valid[ind]:
                    first_idx = first_valid[ind]
                    last_idx = last_valid[ind]
                    
                    if first_idx > 0:
                        result[ind + (slice(None, first_idx),)] = result[ind + (first_idx,)]

                    if last_idx < N - 1:
                        result[ind + (slice(last_idx + 1, None),)] = result[ind + (last_idx,)]

    nan_mask = np.isnan(result)
    valid_mask = ~nan_mask
    
    valid_counts = np.sum(valid_mask, axis=-1)
    interpolate_mask = valid_counts >= 2
    
    if np.any(interpolate_mask):
        flat_result = result.reshape(-1, N)
        flat_nan_mask = nan_mask.reshape(-1, N)
        flat_interpolate_mask = interpolate_mask.reshape(-1)
        
        for i in range(flat_result.shape[0]):
            if not flat_interpolate_mask[i]:
                continue
                
            valid = ~flat_nan_mask[i]
            nan_indices = flat_nan_mask[i]
            
            if np.any(nan_indices):
                flat_result[i, nan_indices] = np.interp(idx[nan_indices],
                                                      idx[valid],
                                                      flat_result[i, valid])
        
        result = flat_result.reshape(shape)

    result = np.moveaxis(result, -1, axis)
    return result

def _interpolate_1d(arr: NDArray, correct_boundaries: bool) -> NDArray:
    """
    Interpolate NaNs for 1-dimensional arrays.
    
    Arguments:
        arr (NDArray): Input array with NaNs.
        correct_boundaries (bool): If True, propagate edge values to NaNs.

    Returns:
        NDArray: Array with NaNs interpolated.
    """
    result = arr.copy()
    nan_mask = np.isnan(result)

    if not np.any(nan_mask):
        return result

    valid_mask = ~nan_mask
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return result

    if len(valid_indices) == 1:
        if correct_boundaries:
            result[:] = result[valid_indices[0]]
        return result

    if correct_boundaries:
        first_valid, last_valid = valid_indices[0], valid_indices[-1]
        result[:first_valid] = result[first_valid]
        result[last_valid + 1:] = result[last_valid]
        nan_mask = np.isnan(result)

    # Only interpolate interior NaNs if not correcting boundaries
    if np.any(nan_mask):
        idx = np.arange(len(result))
        valid_indices = np.where(~nan_mask)[0]
        # Find interior NaNs (between first and last valid indices)
        if correct_boundaries:
            interp_mask = nan_mask
        else:
            if len(valid_indices) >= 2:
                first_valid, last_valid = valid_indices[0], valid_indices[-1]
                interp_mask = np.zeros_like(nan_mask, dtype=bool)
                interp_mask[first_valid+1:last_valid] = nan_mask[first_valid+1:last_valid]
            else:
                interp_mask = np.zeros_like(nan_mask, dtype=bool)
        if np.any(interp_mask):
            interp_indices = np.where(interp_mask)[0]
            result[interp_indices] = np.interp(idx[interp_indices], valid_indices, result[valid_indices])

    return result