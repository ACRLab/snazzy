import json

import numpy as np
import scipy.signal as spsig
from scipy.ndimage import minimum_filter1d
from scipy.stats import zscore


class Trace:
    """Calculates dff and peak data for the resulting trace."""

    def __init__(
        self,
        name,
        activity,
        trim_zscore=0.35,
        dff_strategy="baseline",
        has_transients=False,
        pd_props_path=None,
    ):
        self.name = name
        self.time = activity[:, 0]
        self.active = activity[:, 1]
        self.struct = activity[:, 2]
        self.dff_strategy = dff_strategy
        self.has_transients = has_transients
        self.pd_props_path = pd_props_path  # peak detection props
        self._peak_idxes = None
        self._peak_bounds_indices = None
        self._order_zero_savgol = None
        self.to_add = []
        self.to_remove = []

        self.trim_idx = self.trim_data(trim_zscore)

        self.dff = self.compute_dff()

    @property
    def peak_idxes(self):
        if self._peak_idxes is None:
            if self.pd_props_path and self.pd_props_path.exists():
                try:
                    pd_params = self.get_peak_detection_params()
                except ValueError:
                    pd_params = {}
                self.detect_peaks(**pd_params)
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

    def get_peak_detection_params(self):
        """Reads peak detection params from config file."""
        if self.pd_props_path.exists():
            with open(self.pd_props_path, "r") as f:
                config = json.load(f)
        pd_params = ["mpd", "order0_min", "order1_min", "prominence"]
        if any(param not in config for param in pd_params):
            raise ValueError("Missing params in pd_params file")
        return {param: config[param] for param in pd_params}

    def compute_dff(self, window_size=80):
        """Compute dff for the ratiometric active channel signal."""
        ratiom_signal = self.compute_ratiom_gcamp()
        if self.dff_strategy == "baseline":
            baseline = self.compute_baseline(ratiom_signal, window_size)
        elif self.dff_strategy == "local_minima":
            baseline = minimum_filter1d(ratiom_signal, size=window_size)
        else:
            raise ValueError(
                f"Could not apply the dff_strategy specified: {self.dff_strategy}."
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

    def detect_peaks(
        self, mpd=71, order0_min=0.08, order1_min=0.006, prominence=0.1, max_amp=10
    ):
        self.calculate_peaks(mpd, order0_min, order1_min, prominence, max_amp)

        peak_idxes = self.peak_idxes
        peak_times = self.peak_times

        if self.has_transients:
            avg_ISI = np.average(peak_times[1:] - peak_times[:-1])
            if (peak_times[1] - peak_times[0]) > 2 * avg_ISI:
                peak_idxes = peak_idxes[1:]
                peak_times = peak_times[1:]

        # reconcile the calculated peaks with manually corrected data:
        corrected_peaks = None
        if self.pd_props_path.exists():
            with open(self.pd_props_path, "r") as f:
                config = json.load(f)
                if self.name in config.get("embryos", {}):
                    corrected_peaks = config["embryos"][self.name]
        if corrected_peaks:
            to_add = corrected_peaks["manual_peaks"]
            to_remove = corrected_peaks["manual_remove"]
            wlen = corrected_peaks["wlen"]
            filtered_peaks = [
                p for p in peak_idxes if not any(abs(p - rp) < wlen for rp in to_remove)
            ]
            filtered_add = [
                ap
                for ap in to_add
                if not any(abs(p - ap) < wlen for p in filtered_peaks)
            ]
            peak_idxes = np.array(sorted(filtered_peaks + filtered_add), dtype=np.int64)
            self.to_add = to_add
            self.to_remove = to_remove

        self._peak_idxes = peak_idxes
        peak_times = self.time[peak_idxes]
        # TODO: this return stmt is here only to support the debouncer in ipynb
        # should be removed after the ipynb is rewritten
        return peak_times, peak_idxes

    def calculate_peaks(
        self,
        mpd=71,
        order0_min=0.08,
        order1_min=0.006,
        prominence=0.1,
        max_amp=10,
        extend_true_filters_by=30,
    ):
        """
        Detects peaks using Savitzky-Golay-filtered signal and its derivatives, computed in __init__.
        Partly relies on spsig.find_peaks called on the signal, with parameters mpd (minimum peak distance)
         and order0_min (minimum peak height).
        order1_min sets the minimum first-derivative value, and the second derivative must be <0. These filters
         are stretched out to the right by extend_true_filters_by samples.
        """
        dff = self.dff[: self.trim_idx]
        savgol = self.order_zero_savgol
        order0_idxes = spsig.find_peaks(
            savgol, height=order0_min, prominence=prominence, distance=mpd
        )[0]
        order0_filter = np.zeros(len(savgol), dtype=bool)
        order0_filter[order0_idxes] = True

        savgol1 = spsig.savgol_filter(dff, 21, 4, deriv=1)
        order1_filter = savgol1 > order1_min
        order1_filter = _extend_true_right(order1_filter, extend_true_filters_by)

        savgol2 = spsig.savgol_filter(dff, 21, 4, deriv=2)
        order2_filter = savgol2 < 0
        order2_filter = _extend_true_right(order2_filter, extend_true_filters_by)

        within_limit = dff < max_amp
        joint_filter = np.all(
            [order0_filter, order1_filter, order2_filter, within_limit], axis=0
        )
        self._peak_idxes = np.where(joint_filter)[0]

    def get_first_peak_time(self):
        """Returns the time when the first peak was detected."""
        return self.peak_times[0]

    def trim_data(self, trim_zscore):
        """
        Computes the z score for each Savitzky-Golay-filtered sample, and removes the data 5 samples prior to the
        first sample whose absolute value is greater than the threshold trim_zscore.
        """
        tomato_savgol = spsig.savgol_filter(self.struct, 251, 2, deriv=0)
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
        """Computes properties of each dff peak using spsig.peak_widths."""
        _, _, start_idxs, end_idxs = spsig.peak_widths(
            self.order_zero_savgol, self.peak_idxes, rel_height
        )

        start_idxs = start_idxs.astype(np.int64)
        end_idxs = end_idxs.astype(np.int64)
        self._peak_bounds_indices = np.vstack((start_idxs, end_idxs)).T

        peaks_to_bounds = {
            p: [s, e] for (p, s, e) in zip(self.peak_idxes, start_idxs, end_idxs)
        }

        # reconcile the peak_bounds with manually changed peak widths here:
        # if a peak has associated manual width data, overwrite that peak's bounds
        manual_peak_bounds = None
        if self.pd_props_path.exists():
            with open(self.pd_props_path, "r") as f:
                config = json.load(f)
                if self.name in config.get("embryos", {}):
                    manual_peak_bounds = config["embryos"][self.name].get(
                        "manual_widths", None
                    )

        if manual_peak_bounds:
            # must convert the data from json file to np types:
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
        savgol = spsig.savgol_filter(dff, 21, 4, deriv=0)
        peak_slices = [savgol[x[0] : x[1]] for x in peak_bounds]
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
