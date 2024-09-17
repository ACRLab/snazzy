import numpy as np
import scipy.signal as spsig
from scipy.stats import zscore


class Trace:
    '''Calculates dff and peak data for the resulting trace.'''

    def __init__(self, time, active, struct, trim_zscore=0.35):
        self.time = time
        self.active = active
        self.struct = struct
        self._peak_idxes = None
        self._peak_times = None
        self._peak_intervals = None
        self._peak_amplitudes = None
        self._peak_bounds_indices = None
        self._peak_bounds_time = None
        self._peak_durations = None
        self._peak_aucs = None

        self.trim_idx = self.trim_data(trim_zscore)

        self.dff = self.compute_dff()

    @property
    def peak_idxes(self):
        if self._peak_idxes is None:
            self._detect_peaks()
        return self._peak_idxes

    @property
    def peak_times(self):
        if self._peak_times is None:
            self._detect_peaks()
        return self._peak_times

    @property
    def peak_intervals(self):
        if self._peak_intervals is None:
            self._detect_peaks()
        return self._peak_intervals

    @property
    def peak_amplitudes(self):
        if self._peak_amplitudes is None:
            self._detect_peaks()
        return self._peak_amplitudes

    @property
    def peak_bounds_indices(self):
        if self._peak_bounds_indices is None:
            self.compute_peak_bounds()
        return self._peak_bounds_indices

    @property
    def peak_durations(self):
        if self._peak_bounds_time is None:
            self.compute_peak_bounds()
        return [end - start for (start, end) in self._peak_bounds_time]

    @property
    def peak_rise_times(self):
        if self._peak_bounds_time is None:
            self.compute_peak_bounds()
        start_times = self._peak_bounds_time[:, 0]
        return self.peak_times - start_times

    @property
    def peak_decay_times(self):
        if self._peak_bounds_time is None:
            self.compute_peak_bounds()
        end_times = self._peak_bounds_time[:, 1]
        return end_times - self.peak_times

    @property
    def peak_aucs(self):
        if self._peak_aucs is None:
            self.compute_peak_aucs_from_bounds()
        return self._peak_aucs

    @property
    def rms(self):
        return np.sqrt(np.mean((self.dff[:self.trim_idx])**2))

    def compute_dff(self):
        '''Compute dff for the ratiometric active channel signal.'''
        ratiom_signal = self.compute_ratiom_gcamp()
        baseline = self.compute_baseline(ratiom_signal)
        return (ratiom_signal-baseline)/baseline

    def compute_ratiom_gcamp(self):
        '''Computes the ratiometric GCaMP signal by dividing the raw GCaMP 
        signal by the tdTomato signal.'''
        return self.active/self.struct

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

    def _detect_peaks(self, mpd=71, order0_min=0.08, order1_min=0.006, extend_true_filters_by=30, has_transients=True):
        '''
        Detects peaks using Savitzky-Golay-filtered signal and its derivatives, computed in __init__.
        Partly relies on spsig.find_peaks called on the signal, with parameters mpd (minimum peak distance)
         and order0_min (minimum peak height).
        order1_min sets the minimum first-derivative value, and the second derivative must be <0. These filters
         are stretched out to the right by extend_true_filters_by samples. 
        '''
        dff = self.dff[:self.trim_idx]
        savgol = spsig.savgol_filter(dff, 21, 4, deriv=0)
        order0_idxes = spsig.find_peaks(
            savgol, height=order0_min, distance=mpd)[0]
        order0_filter = np.zeros(len(savgol), dtype=bool)
        order0_filter[order0_idxes] = True

        savgol1 = spsig.savgol_filter(dff, 21, 4, deriv=1)
        order1_filter = savgol1 > order1_min
        order1_filter = _extend_true_right(
            order1_filter, extend_true_filters_by)

        savgol2 = spsig.savgol_filter(dff, 21, 4, deriv=2)
        order2_filter = savgol2 < 0
        order2_filter = _extend_true_right(
            order2_filter, extend_true_filters_by)

        joint_filter = np.all(
            [order0_filter, order1_filter, order2_filter], axis=0)
        peak_idxes = np.where(joint_filter)[0]
        if peak_idxes.size == 0:
            raise ValueError("No peaks found, cannot derive trace metrics.")
        peak_times = self.time[peak_idxes]

        if has_transients:
            avg_ISI = np.average(peak_times[1:] - peak_times[:-1])
            if (peak_times[1] - peak_times[0]) > 2 * avg_ISI:
                peak_idxes = peak_idxes[1:]
                peak_times = peak_times[1:]

        self._peak_idxes = peak_idxes
        self._peak_times = peak_times
        self._peak_intervals = np.diff(peak_times)
        self._peak_amplitudes = savgol[peak_idxes]

    def get_first_peak_time(self):
        '''Returns the time when the first peak was detected.'''
        return self.peak_times[0]

    def trim_data(self, trim_zscore):
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
        if len(trim_points) == 0:
            trim_idx = len(self.time) - 5
        else:
            trim_idx = trim_points[0] - 5
        return trim_idx

    def compute_peak_bounds(self, rel_height=0.92):
        '''Computes properties of each dff peak using spsig.peak_widths.'''
        dff = self.dff[:self.trim_idx]
        savgol = spsig.savgol_filter(dff, 21, 4, deriv=0)
        _, _, start_idxs, end_idxs = spsig.peak_widths(
            savgol, self.peak_idxes, rel_height)

        X = np.arange(len(self.time))
        start_times = np.interp(start_idxs, X, self.time)
        end_times = np.interp(end_idxs, X, self.time)
        # combines two 1D nparrs of shape (n) into a 2D nparr of shape (n,2)
        bound_times = np.vstack((start_times, end_times)).T

        start_idxs = start_idxs.astype(np.int64)
        end_idxs = end_idxs.astype(np.int64)
        self._peak_bounds_indices = np.vstack((start_idxs, end_idxs)).T
        self._peak_bounds_time = bound_times

    def get_peak_slices_from_bounds(self):
        dff = self.dff[:self.trim_idx]
        peak_bounds = self.peak_bounds_indices
        savgol = spsig.savgol_filter(dff, 21, 4, deriv=0)
        peak_slices = [savgol[x[0]:x[1]] for x in peak_bounds]
        time_slices = [self.time[x[0]:x[1]] for x in peak_bounds]
        return list(zip(peak_slices, time_slices))

    def compute_peak_aucs_from_bounds(self):
        peak_time_slices = self.get_peak_slices_from_bounds()
        peak_aucs = np.asarray([np.trapz(pslice*100, tslice)
                               for pslice, tslice in peak_time_slices])

        self._peak_aucs = peak_aucs


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
