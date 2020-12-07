"""
    Finds high-frequency bursts in a background low-frequency signal.
    
    This is an example of a more advanced Python script for signal processing.
    The details of the algorithm here aren't important.  What matters is
    the workflow: we aquire data from LabVIEW, send it to Python, process it
    and then return the analysis to LabVIEW.
    
    In this example, we send a waveform to Python as a 1D array of doubles.
    The regions where high-frequency bursts occur are returned in a 2D array
    with shape (N, 2).  This is effectively a list of the start/stop points
    for the regions detected by Python.
"""

# Imports.  This module will use the numpy and scipy packages.
import sys
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy import hanning, ndimage

if sys.version_info[0] > 2:
    xrange = range

def find_bursts(signal, sampling_rate=10,
                        freq_shift_threshold=0.125, 
                        frame_size=64,
                        overlap_size=None):
    """ Detect high-frequency bursts in a signal.
    
    This function takes one required argument, the signal to process (as a 1D
    array of floats or doubles).  Optional arguments may be supplied for
    sampling rate, frequency threshold, or other tuning parameters.
    
    Returns a 2D array with shape (number of regions, 2).
    """
    if not overlap_size:
        overlap_size = frame_size // 2

    t, f, stft_mag = stft(signal, frame_size, sampling_rate, overlap=None)
    f_max = f[np.argmax(np.abs(stft_mag), axis=0)]

    mask = np.zeros_like(f_max, dtype='bool')
    mask[1:] = np.abs(np.diff(f_max)) > freq_shift_threshold

    # Denoise the mask : remove spikes between the normal and event state
    # remove small spikes from false to True
    denoised_mask = ndimage.binary_opening(mask, [True]*3)

    # remove small spikes from false to True
    denoised_mask = ndimage.binary_opening(mask, [True]*3)

    # remove small drops from true to false
    inv_mask = ~denoised_mask
    denoised_mask = ~(ndimage.binary_opening(inv_mask, [True]*5))

    runs = arg_find_runs(denoised_mask, order='flat')
    if denoised_mask[0]:
        true_runs = runs[::2, :]
    else:
        true_runs = runs[1::2, :]

    # Throw away very short duration events
    run_mask = (np.diff(true_runs) > 4).ravel()
    significant_runs = true_runs[run_mask, :].reshape(-1, 2)

    # Convert to start/stop times, using the sampling rate defined above
    freq_shift_regions = np.empty((len(significant_runs), 2), dtype='f')
    for idx, run in enumerate(significant_runs):
        start, stop = run
        freq_shift_regions[idx] = t[start], t[stop-1]
        
    # Return data to LabVIEW
    return freq_shift_regions
    
    
    




# --- Support functions ------------------------------------------------------

def right_shift(ary, newval):
    """ Returns a right-shifted version of *ary* with *newval* inserted on the
    left. """
    return np.concatenate([[newval], ary[:-1]])


def left_shift(ary, newval):
    """ Returns a left-shifted version of *ary* with *newval* inserted on the
    right. """
    return np.concatenate([ary[1:], [newval]])
    
def arg_find_runs(int_array, order='ascending'):
    """ Return an (N, 2) int array giving slices of runs.
    Parameters
    ----------
    int_array : int array_like
    order : 'ascending', 'descending', or 'flat'
        Find runs of ascending integers (1, 2, 3), descending integers (3, 2,
        1), or constant (1, 1, 1).
    Returns
    -------
    slices : (N, 2) int array
        Start and (exclusive) stop indices into the original array of the
        contiguous runs.
    """
    if len(int_array) == 0:
        return np.empty((0, 2), dtype=int)
    if order == 'ascending':
        increment = 1
    elif order == 'descending':
        increment = -1
    else:
        increment = 0
    rshifted = right_shift(int_array, int_array[0]-increment).view(np.ndarray)
    start_indices = np.concatenate([
        [0],
        np.nonzero(int_array - (rshifted + increment))[0],
    ])
    end_indices = left_shift(start_indices, len(int_array))
    return np.column_stack([start_indices, end_indices])
    
def stft(data, frame_size, sampling_rate, overlap=None):
    if not overlap:
        overlap = frame_size // 2
    #print data, frame_size, sampling_rate, overlap
    stride = frame_size - overlap
    x = np.arange(0, len(data)-frame_size, stride) * 1.0/sampling_rate
    f = rfftfreq(frame_size, 1.0/sampling_rate)
    w_kern = hanning(len(f))
    out = np.abs(
        np.array([rfft(data[i:i+frame_size]) * w_kern
                  for i in xrange(0, len(data)-frame_size, stride)])
    )
    return x, f, out.T
