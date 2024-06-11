import numpy as np
import scipy.signal as spsig
from scipy.stats import zscore


class Trace:
    '''Calculates dff and peak data for the resulting trace.'''

    def __init__(self, time, active, struct, trim_data=True, trim_zscore=0.35):
        self.time = time
        self.active = active
        self.struct = struct

        if trim_data:
            self.trim_idx = self.trim_data(trim_zscore)

        self.dff = self.compute_dff()
        # Compute Savitzky-Golay-filtered signal and its 1st and 2nd derivatives
        self.savgol = spsig.savgol_filter(self.dff, 21, 4, deriv=0)
        self.savgol1 = spsig.savgol_filter(self.dff, 21, 4, deriv=1)
        self.savgol2 = spsig.savgol_filter(self.dff, 21, 4, deriv=2)

    def compute_dff(self):
        '''Compute dff for the ratiometric active channel signal.'''
        ratiom_signal = self.compute_ratiom_gcamp()
        baseline = self.compute_baseline(ratiom_signal)
        return self._dff(ratiom_signal, baseline)

    def compute_ratiom_gcamp(self):
        '''Computes the ratiometric GCaMP signal by dividing the raw GCaMP 
        signal by the tdTomato signal.'''
        if self.trim_idx:
            active = self.active[:self.trim_idx]
            struct = self.struct[:self.trim_idx]
            return active/struct
        return self.active/self.struct

    def _dff(self, signal, baseline):
        '''Helper function to compute deltaF/F given signal and baseline.'''
        return (signal-baseline)/baseline

    def reflect_edges(self, signal, window_size=160):
        '''Reflects edges so we can fit windows of size window_size for the
        entire signal.'''
        half = window_size//2
        return np.concatenate((signal[:half], signal, signal[-half:]))

    def compute_baseline(self, signal, window_size=160, n_bins=20):
        '''Compute baseline for each sliding window by dividing up the signal
        into n_bins amplitude bins and taking the mean of the bin with the most
        samples.

        This assumes that PaSNA peaks are sparse.
        To handle edge cases, both edges are reflected.
        '''
        expanded_signal = self.reflect_edges(signal, window_size)

        baseline = np.zeros_like(signal)
        for i, _ in enumerate(signal):
            window = expanded_signal[i:i+window_size]
            counts, bins = np.histogram(window, bins=n_bins)
            mode_bin_idx = np.argmax(counts)
            mode_bin_mask = np.logical_and(
                window > bins[mode_bin_idx], window <= bins[mode_bin_idx+1])
            window_baseline = np.mean(window[mode_bin_mask])
            baseline[i] = window_baseline
        return baseline

    def detect_peaks(self, mpd=71, order0_min=0.06, order1_min=0.006, extend_true_filters_by=30):
        '''
        Detects peaks using Savitzky-Golay-filtered signal and its derivatives, computed in __init__.
        Partly relies on spsig.find_peaks called on the signal, with parameters mpd (minimum peak distance)
         and order0_min (minimum peak height).
        order1_min sets the minimum first-derivative value, and the second derivative must be <0. These filters
         are stretched out to the right by extend_true_filters_by samples. 
        '''
        order0_idxes = spsig.find_peaks(
            self.savgol, height=order0_min, distance=mpd)[0]
        order0_filter = np.zeros(len(self.savgol), dtype=bool)
        order0_filter[order0_idxes] = True

        order1_filter = self.savgol1 > order1_min
        order1_filter = _extend_true_right(
            order1_filter, extend_true_filters_by)

        order2_filter = self.savgol2 < 0
        order2_filter = _extend_true_right(
            order2_filter, extend_true_filters_by)

        joint_filter = np.all(
            [order0_filter, order1_filter, order2_filter], axis=0)
        peak_idxes = np.where(joint_filter)[0]
        peak_times = self.time[peak_idxes]

        self.peak_idxes = peak_idxes
        self.peak_times = peak_times
        self.peak_intervals = np.diff(peak_times)
        self.peak_amplitudes = self.savgol[peak_idxes]

        return self.peak_times

    def get_first_peak_time(self):
        '''Returns the time when the first peak was detected.'''
        if not any(self.peak_times):
            self.detect_peaks()
        return self.peak_times[0]

    def trim_data(self, trim_zscore=5):
        '''
        Computes the z score for each Savitzky-Golay-filtered sample, and removes the data 5 samples prior to the 
        first sample whose absolute value is greater than the threshold trim_zscore. 
        '''
        tomato_savgol = spsig.savgol_filter(
            self.struct, 251, 2, deriv=0)
        zscored_tomato = zscore(tomato_savgol)
        zscored_tomato -= self.compute_baseline(zscored_tomato, window_size=51)

        trim_points = np.where(np.abs(zscored_tomato) > trim_zscore)[0]
        # Trim 5 timepoints before
        trim_idx = trim_points[0]-5 if len(trim_points) > 0 else None
        return trim_idx


def _extend_true_right(bool_array, n_right):
    '''
    Helper function that takes in a boolean array and extends each stretch of True values by n_right indices.

    Example:
    >> extend_true_right([False, True, True, False, False, True, False], 1)
    returns:             [False, True, True, True,  False, True, True]
    '''
    extended = np.zeros_like(bool_array, dtype=bool)
    for i in range(len(bool_array)):
        if bool_array[i] == True:
            rb = i+n_right
            rb = min(rb, len(bool_array))
            extended[i:rb] = True
    return extended
