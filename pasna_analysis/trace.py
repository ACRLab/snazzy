import numpy as np
import scipy.signal as spsig
from scipy.ndimage import minimum_filter1d
from scipy.stats import zscore

from pasna_analysis import Config


class Trace:
    """Calculates dff and peak data for the resulting trace."""

    def __init__(
        self,
        name,
        activity,
        config: Config,
        fs=1 / 6,
    ):
        self.name = name
        self.time = activity[:, 0]
        self.active = activity[:, 1]
        self.struct = activity[:, 2]
        self.config = config
        self.pd_params = config.get_pd_params()

        # list of peaks that were manually added / removed:
        self.to_add = []
        self.to_remove = []
        self.fs = fs

        self._peak_idxes = None
        self._peak_bounds_indices = None
        self._order_zero_savgol = None
        self.filtered_dff = None

        self.trim_idx = self.get_trim_index()
        self.dff = self.compute_dff()

    @property
    def peak_idxes(self):
        if self._peak_idxes is None:
            if "freq" in self.pd_params:
                self.detect_peaks(self.pd_params["freq"])
            else:
                self.detect_peaks()
        return self._peak_idxes

    @peak_idxes.setter
    def peak_idxes(self, peak_idxes):
        self._peak_idxes = peak_idxes

    @property
    def peak_times(self):
        return self.time[self.peak_idxes]

    @property
    def peak_intervals(self):
        return np.diff(self.peak_times)

    @property
    def peak_amplitudes(self):
        return self.dff[self.peak_idxes]

    @property
    def peak_bounds_indices(self):
        if self._peak_bounds_indices is None:
            self.compute_peak_bounds()
        return self._peak_bounds_indices

    @peak_bounds_indices.setter
    def peak_bounds_indices(self, peak_bounds):
        self._peak_bounds_indices = peak_bounds

    @property
    def peak_bounds_times(self):
        return self.get_peak_bounds_times()

    @property
    def peak_durations(self):
        return [end - start for (start, end) in self.peak_bounds_times]

    @property
    def peak_rise_times(self):
        try:
            start_times = self.peak_bounds_times[:, 0]
        except IndexError:
            return []
        return self.peak_times - start_times

    @property
    def peak_decay_times(self):
        try:
            end_times = self.peak_bounds_times[:, 1]
        except IndexError:
            return []
        return end_times - self.peak_times

    @property
    def peak_aucs(self):
        return self.get_peak_aucs_from_bounds()

    @property
    def rms(self):
        return np.sqrt(np.mean((self.dff[: self.trim_idx]) ** 2))

    @property
    def order_zero_savgol(self):
        if self._order_zero_savgol is None:
            self._order_zero_savgol = spsig.savgol_filter(
                self.dff[: self.trim_idx], 21, 4, deriv=0
            )
        return self._order_zero_savgol

    def compute_dff(self, window_size=80):
        """Compute dff for the ratiometric active channel signal."""
        ratiom_signal = self.compute_ratiom_gcamp()
        dff_strategy = self.pd_params.get("dff_strategy", "")
        if dff_strategy == "baseline":
            baseline = self.compute_baseline(ratiom_signal, window_size)
        elif dff_strategy == "local_minima":
            baseline = minimum_filter1d(ratiom_signal, size=window_size)
        else:
            raise ValueError(
                f"Could not apply the dff_strategy specified: {dff_strategy}."
            )
        return (ratiom_signal - baseline) / baseline

    def compute_ratiom_gcamp(self):
        """Computes the ratiometric GCaMP signal by dividing the raw GCaMP
        signal by the tdTomato signal."""
        return self.active / self.struct

    def reflect_edges(self, signal, window_size=160):
        """Reflects edges so we can fit windows of size window_size for the
        entire signal."""
        half = window_size // 2
        return np.concatenate((signal[:half], signal, signal[-half:]))

    def compute_baseline(self, signal, window_size=160, n_bins=64):
        """Compute baseline for each sliding window by dividing up the signal
        into n_bins amplitude bins and taking the mean of the bin with the most
        samples.

        This assumes that PaSNA peaks are sparse.
        To handle edges, both edges are reflected.
        """
        expanded_signal = self.reflect_edges(signal, window_size)

        baseline = np.zeros_like(signal)
        rng = np.min(signal), np.max(signal)
        counts, bins = np.histogram(
            expanded_signal[:window_size], bins=n_bins, range=rng
        )

        window = expanded_signal[:window_size]
        mode_bin_idx = np.argmax(counts)
        mode_bin_mask = np.logical_and(
            window > bins[mode_bin_idx], window <= bins[mode_bin_idx + 1]
        )
        baseline[0] = np.mean(window[mode_bin_mask])

        for i in range(1, len(signal)):
            window = expanded_signal[i : i + window_size]
            out_point = expanded_signal[i - 1]
            in_point = expanded_signal[i + window_size - 1]

            out_bin = np.searchsorted(bins, out_point, side="right") - 1
            in_bin = np.searchsorted(bins, in_point, side="left") - 1

            if 0 <= out_bin < len(counts):
                counts[out_bin] -= 1
            if 0 <= in_bin < len(counts):
                counts[in_bin] += 1

            mode_bin_idx = np.argmax(counts)
            mode_bin_mask = np.logical_and(
                window > bins[mode_bin_idx], window <= bins[mode_bin_idx + 1]
            )
            baseline[i] = np.mean(window[mode_bin_mask])

        return baseline

    def process_peaks(self, stages):
        """Applies a series of stages to process peak data from ss.find_peaks.

        Each stage is a callable that takes a list of peaks and a params dict."""
        for stage_func, params in stages:
            stage_func(params)

    def remove_transients(self, params):
        """Transients are very early bouts that result in a first peak that
        happens way before other peaks."""
        if not self.pd_params.get("has_transients", False):
            return
        ISI_factor = params.get("ISI_factor", 4)
        # average inter-spike interval
        peak_times = self.time[self._peak_idxes]
        if len(peak_times) < 2:
            print("Cannot remove transients, not enough peaks.")
            return
        avg_ISI = np.average(peak_times[1:] - peak_times[:-1])
        if (peak_times[1] - peak_times[0]) > ISI_factor * avg_ISI:
            self._peak_idxes = np.array(self._peak_idxes[1:])

    def apply_low_threshold(self, params):
        """Removes all peaks below a percentage of the max amplitude."""
        max_peak = max(self.peak_amplitudes)
        threshold = params.get("low_amp_threshold", 0.1)
        self._peak_idxes = np.array(
            [p for p in self._peak_idxes if self.dff[p] > threshold * max_peak]
        )

    def reconcile_manual_peaks(self, params):
        """Reconcile the calculated peaks with manually corrected data.

        The manual data is written at `self.pd_props_path`, through the GUI."""
        if not self.config:
            return
        corrected_peaks = None

        corrected_peaks = self.config.get_corrected_peaks(self.name)
        if corrected_peaks:
            to_add = corrected_peaks["manual_peaks"]
            to_remove = corrected_peaks["manual_remove"]
            wlen = corrected_peaks["wlen"]
            filtered_peaks = [
                p
                for p in self._peak_idxes
                if not any(abs(p - rp) < wlen for rp in to_remove)
            ]
            filtered_add = [
                ap
                for ap in to_add
                if not any(abs(p - ap) < wlen for p in filtered_peaks)
            ]
            self.to_add = to_add
            self.to_remove = to_remove
            self._peak_idxes = np.array(
                sorted(filtered_peaks + filtered_add), dtype=np.int64
            )

    def detect_peaks(self, freq=0.0025):
        self._peak_idxes, filtered_dff = self.calculate_peaks(freq_cutoff=freq)
        self.filtered_dff = filtered_dff

        stages = [
            (self.remove_transients, {}),
            (self.apply_low_threshold, {"low_amp_threshold": 0.05}),
            (self.reconcile_manual_peaks, {}),
        ]

        self.process_peaks(stages)

    def detect_oscillations(self, after_ep=2, offset=10):
        mask = np.zeros_like(self.dff, dtype=np.bool_)
        peak_bounds = self.peak_bounds_indices
        for s, e in peak_bounds:
            mask[s : e + offset] = 1
        # oscillations only happen in phase 2, which should be at least after ep 2:
        if len(self.peak_idxes) <= after_ep:
            return None
        p2_start = self.peak_idxes[after_ep]
        dff = self.dff.copy()
        dff[mask] = 0
        dff = dff[p2_start : self.trim_idx]
        order0_idxes = spsig.find_peaks(dff, height=0.1, distance=5, prominence=0.07)[0]

        return order0_idxes + p2_start

    def get_filtered_signal(self, freq_cutoff):
        N = len(self.dff[: self.trim_idx])
        freqs = np.fft.rfftfreq(N, 1 / self.fs)
        fft = np.fft.rfft(self.dff[: self.trim_idx])
        mask = freqs < freq_cutoff
        filtered_fft = fft * mask
        filtered_signal = np.fft.irfft(filtered_fft, n=N)

        return filtered_signal

    def filter_peaks_by_local_context(
        self, signal, peak_indices, window_size=300, value=75, method="percentile"
    ):
        """
        Filter pre-detected peaks by comparing their height to a local threshold.

        Parameters:
            signal (np.ndarray): Original signal.
            peak_indices (np.ndarray): Indices of peaks (e.g., from find_peaks).
            window_size (int): Size of the window to determine local threshold.
            value (int): Factor to determine local threshold.
            method ('mean' | 'median' | 'percentile'): How to calculate the threshold.

        Returns:
            filtered_peaks (np.ndarray): Indices of peaks that passed local thresholding.
        """
        filtered_peaks = []

        for i in peak_indices:
            if signal[i] > self.local_threshold(signal, i, window_size, value, method):
                filtered_peaks.append(i)

        return np.array(sorted(filtered_peaks))

    def local_threshold(self, signal, idx, window_size, value, method):
        half_win = window_size // 2

        start = max(0, idx - half_win)
        end = min(len(signal), idx + half_win)
        window = signal[start:end]

        if method == "mean":
            local_thresh = value * np.mean(np.abs(window))
        elif method == "median":
            local_thresh = value * np.median(np.abs(window))
        elif method == "percentile":
            local_thresh = np.percentile(np.abs(window), value)
        else:
            raise ValueError("Method must be 'mean', 'median', or 'percentile'.")

        return local_thresh

    def port_peaks(self, peaks, target_signal, search_window=30, peak_height_thres=0.7):
        """Changes peaks index to the highest peak amplitude on a target signal."""
        local_peak_indices = []

        for idx in peaks:
            left = max(0, idx - search_window)
            right = min(len(target_signal), idx + search_window)
            window = target_signal[left:right]
            if len(window) == 0:
                continue

            local_peaks, peak_data = spsig.find_peaks(window, height=(None, None))
            local_peak_heights = peak_data["peak_heights"]
            max_peak = np.max(local_peak_heights)

            leftmost_peak = next(
                p
                for (p, ph) in zip(local_peaks, local_peak_heights)
                if ph >= peak_height_thres * max_peak
            )
            local_peak_indices.append(left + leftmost_peak)

        return np.array(local_peak_indices)

    def calculate_peaks(self, freq_cutoff):
        """
        Detect peaks based on the low-passed iFFT signal. Peaks found in the
        freq domain are used as anchor points in the actual signal to determine
        peak indices.
        """
        filtered_dff = self.get_filtered_signal(freq_cutoff)

        peak_indices, _ = spsig.find_peaks(filtered_dff, height=0.04, prominence=0.03)

        local_peak_indices = self.port_peaks(peak_indices, self.dff[: self.trim_idx])

        peaks = self.filter_peaks_by_local_context(self.dff, local_peak_indices)

        return np.array(peaks), filtered_dff

    def get_first_peak_time(self):
        """Returns the time when the first peak was detected."""
        return self.peak_times[0]

    def get_trim_index(self):
        """Try to return the trim index from config, otherwise calculates it."""
        corrected_data = self.config.get_corrected_peaks(self.name)
        if corrected_data:
            manual_trim_idx = corrected_data.get("manual_trim_idx", -1)
            if manual_trim_idx != -1:
                return corrected_data.get("manual_trim_idx")
        trim_zscore = self.pd_params.get("trim_zscore", 0.35)
        return self.trim_data(trim_zscore)

    def trim_data(self, trim_zscore):
        """Computes the z score for each Savitzky-Golay-filtered sample, and removes all points after reaching `trim_zscore`."""
        tomato_savgol = spsig.savgol_filter(self.struct, 21, 2, deriv=0)
        zscored_tomato = zscore(tomato_savgol)
        zscored_tomato -= self.compute_baseline(zscored_tomato, window_size=10)

        trim_points = np.where(np.abs(zscored_tomato) > trim_zscore)[0]
        if len(trim_points) == 0:
            trim_idx = len(self.time)
        else:
            trim_idx = trim_points[0]
        return trim_idx

    def compute_peak_bounds(self, rel_height=0.92):
        """Computes properties of each dff peak using spsig.peak_widths."""
        dff = self.dff[: self.trim_idx].copy()
        # make sure that the peak we pass is not bound by local points
        dff[self.peak_idxes] += 3
        _, _, start_idxs, end_idxs = spsig.peak_widths(
            dff, self.peak_idxes, rel_height, wlen=150
        )

        start_idxs = start_idxs.astype(np.int64)
        end_idxs = end_idxs.astype(np.int64)
        self._peak_bounds_indices = np.vstack((start_idxs, end_idxs)).T

        peaks_to_bounds = {
            p: [s, e] for (p, s, e) in zip(self.peak_idxes, start_idxs, end_idxs)
        }

        # reconcile the peak_bounds with manually changed peak widths here:
        # if a peak has associated manual width data, overwrite that peak's bounds
        corrected_peaks = self.config.get_corrected_peaks(self.name)

        if corrected_peaks:
            manual_peak_bounds = corrected_peaks.get("manual_widths", None)

            if manual_peak_bounds:
                # must convert the data from json file to np type:
                manual_peak_bounds = {
                    np.int64(k): [np.int64(n) for n in v]
                    for k, v in manual_peak_bounds.items()
                }

                for peak in manual_peak_bounds:
                    peaks_to_bounds[peak] = manual_peak_bounds[peak]

        self._peak_bounds_indices = np.array(list(peaks_to_bounds.values()))

    def get_peak_bounds_times(self):
        """Returns times at peak boundaries as a 2D nparray of shape (N, 2),
        where the inner dim represents [start_time, end_time] and N is the
        number of peaks."""
        try:
            start_idxs = self.peak_bounds_indices[:, 0]
            end_idxs = self.peak_bounds_indices[:, 1]
        except IndexError:
            return np.array([])

        bound_times = np.vstack((self.time[start_idxs], self.time[end_idxs])).T
        return bound_times

    def get_peak_slices_from_bounds(self):
        dff = self.dff[: self.trim_idx]
        peak_bounds = self.peak_bounds_indices
        peak_slices = [dff[x[0] : x[1]] for x in peak_bounds]
        time_slices = [self.time[x[0] : x[1]] for x in peak_bounds]
        return list(zip(peak_slices, time_slices))

    def get_peak_aucs_from_bounds(self):
        peak_time_slices = self.get_peak_slices_from_bounds()
        peak_aucs = np.asarray(
            [np.trapz(pslice * 100, tslice) for pslice, tslice in peak_time_slices]
        )
        return peak_aucs

    def compute_local_peaks(self, height=0.02, prominence=0.01):
        """Counts local peaks that happen within an episode."""
        peak_bounds = self.peak_bounds_indices

        local_peaks = []
        for s, e in peak_bounds:
            if e >= self.trim_idx:
                break
            local_trace = self.dff[s:e]
            peak_indices, _ = spsig.find_peaks(
                local_trace, height=height, prominence=prominence
            )
            local_peaks.append(len(peak_indices))
        return local_peaks

    def compute_all_local_peaks(self, split_idx, height=0.02, prominence=0.01):
        """Counts number of local peaks before and after the `split_idx`."""
        first_peak = self.peak_idxes[0]
        dff = self.dff[: self.trim_idx]

        lp_before, _ = spsig.find_peaks(
            dff[first_peak:split_idx], height=height, prominence=prominence
        )
        lp_after, _ = spsig.find_peaks(
            dff[split_idx:], height=height, prominence=prominence
        )

        return (len(lp_before), len(lp_after))

    def stft(self, fs=1 / 6, fft_size=600, noverlap=None, duration=3600):
        left_pad = 150
        if noverlap is None:
            noverlap = 3 * (fft_size / 4)

        dff = np.zeros(duration)

        try:
            start = self.peak_bounds_indices[0][0] - left_pad
        except IndexError:
            print(f"Cannot find onset for {self.name}. Cannot calculate STFT.")
            return

        if start < 0:
            print(f"{self.name} onset happened too soon. Cannot calculate STFT.")
            return

        end = self.trim_idx
        # the num of points used to calculate stft cannot exceed `duration`:
        if self.trim_idx - start > duration:
            end = start + duration
        dff[: end - start] = self.dff[start:end]
        return spsig.stft(dff, fs, nperseg=fft_size, noverlap=noverlap, nfft=fft_size)


def _extend_true_right(bool_array, n_right):
    """
    Helper function that takes in a boolean array and extends each stretch of True values by n_right indices.

    Example:
    >> extend_true_right([False, True, True, False, False, True, False], 1)
    returns:             [False, True, True, True,  False, True, True]
    """
    extended = np.zeros_like(bool_array, dtype=bool)
    for i, bool_val in enumerate(bool_array):
        if bool_val:
            extended[i : i + n_right] = True
    return extended
