import bisect

import numpy as np
import scipy.signal as spsig
from scipy.stats import zscore

from pasna_analysis import Config, TracePhases


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
        self.exp_params = config.get_exp_params()

        # list of peaks that were manually added / removed:
        self.to_add = []
        self.to_remove = []
        self.fs = fs

        self._peak_idxes = None
        self._peak_bounds_indices = None
        self._order_zero_savgol = None
        self.filtered_dff = None
        self.dsna_start = None

        self.trim_idx = self.get_trim_index()
        self.dff = self.compute_dff()
        self.aligned_time, self.aligned_dff = self.preprocess_dff()

    @property
    def peak_idxes(self):
        if self._peak_idxes is None:
            if "freq" in self.pd_params:
                self.detect_peaks(self.pd_params["freq"])
            else:
                self.detect_peaks()
        if self.exp_params.get("has_dsna", False):
            filtered_peaks = [pi for pi in self._peak_idxes if pi < self.dsna_start]
            return np.array(filtered_peaks)
        return self._peak_idxes

    @peak_idxes.setter
    def peak_idxes(self, peak_idxes):
        self._peak_idxes = peak_idxes

    @property
    def peak_times(self):
        if len(self.peak_idxes) == 0:
            return []
        return self.time[self.peak_idxes]

    @property
    def peak_intervals(self):
        return np.diff(self.peak_times)

    @property
    def peak_amplitudes(self):
        if len(self.peak_idxes) == 0:
            return []
        return self.dff[self.peak_idxes]

    @property
    def peak_bounds_indices(self):
        if "peak_width" in self.pd_params:
            self._peak_bounds_indices = self.compute_peak_bounds(
                self.pd_params["peak_width"]
            )
        else:
            self._peak_bounds_indices = self.compute_peak_bounds()
        return self._peak_bounds_indices

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

    def get_all_peak_idxes(self):
        if self._peak_idxes is None:
            if "freq" in self.pd_params:
                self.detect_peaks(self.pd_params["freq"])
            else:
                self.detect_peaks()
        return self._peak_idxes

    def compute_dff(self):
        """Compute dff for the ratiometric active channel signal."""
        default_ws = self.config.default_params.get("baseline_window_size")
        window_size = self.pd_params.get("baseline_window_size", default_ws)

        ratiom_signal = self.compute_ratiom_gcamp()
        dff_strategy = self.pd_params.get("dff_strategy", "")
        if dff_strategy == "baseline":
            baseline = self.compute_baseline(ratiom_signal, window_size)
        elif dff_strategy == "local_minima":
            baseline = self.average_n_lowest_window(
                ratiom_signal, window_size, n_lowest=11
            )
        else:
            raise ValueError(
                f"Could not apply the dff_strategy specified: {dff_strategy}."
            )
        return (ratiom_signal - baseline) / baseline

    def preprocess_dff(self, duration=3600, onset_pad=300):
        """Adjust dff to a given duration.

        If embryo hatches, only data from onset to hatching are included.
        If activity after onset does not continue for the entire duration,
        the dff is padded.
        The resulting time and dff have the same length.

        Parameters:
            emb (Embryo):
                The embryo to process.
            duration (int):
                Target num of frames for time and dff.
            onset_pad (int):
                Num of frames to include before onset.

        Returns:
            time_processed (arr):
                Trimmed/extended time.
            dff_processed (arr):
                Trimmed/extended dff.
        """

        time = self.time / 60  # covert to mins
        dff = self.dff

        # trim dff from onset to hatching
        if self.peak_bounds_indices is not None and len(self.peak_bounds_indices) > 0:
            onset = self.peak_bounds_indices[0][0]
        else:
            onset = 0
        if onset > onset_pad:
            start_index = onset - onset_pad
        else:
            start_index = onset
        dff = dff[start_index : self.trim_idx]

        # trim/extend dff to duration
        if duration > len(dff):
            pad_size = duration - len(dff)
            dff_processed = np.array(list(dff) + [0] * pad_size, dtype=object)
        else:
            dff_processed = dff[0:duration]

        increment = time[1] - time[0]
        time_processed = np.arange(0, duration / 10, increment)

        return time_processed, dff_processed

    def compute_ratiom_gcamp(self):
        """Computes the ratiometric GCaMP signal by dividing the raw GCaMP
        signal by the tdTomato signal."""
        return self.active / self.struct

    def reflect_edges(self, signal, window_size=160):
        """Reflects edges so we can fit windows of size window_size for the
        entire signal."""
        half = window_size // 2
        return np.concatenate((signal[:half], signal, signal[-half:]))

    @staticmethod
    def average_n_lowest_window(arr, window_size, n_lowest):
        """Compute the average of the N lowest values for all elements of the array.

        Values at the start and end of the array are reflected.
        This function uses a sorted sliding window to keep track of the lowest
        elements efficiently.

        Parameters:
            arr(list[int]):
                Input that will be filtered.
            window_size(int):
                Number of elements used to look for the lowest values. The window is
                centered at the corresponding element.
            n_lowest(int):
                Number of lowest elements to pick.

        Returns:
            averages(ndarray):
                List of same size as `arr` with the average of n_lowest values of a
                window centered at each element.
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if n_lowest > window_size:
            raise ValueError("n_lowest cannot be greater than window_size")

        half_window = window_size // 2
        padded = np.pad(arr, pad_width=half_window, mode="reflect")
        result = np.empty_like(arr, dtype=float)

        sorted_window = sorted(padded[:window_size])
        result[0] = np.mean(sorted_window[:n_lowest])

        for i in range(1, len(arr)):
            outgoing = padded[i - 1]
            incoming = padded[i + window_size - 1]

            idx = bisect.bisect_left(sorted_window, outgoing)
            sorted_window.pop(idx)

            bisect.insort(sorted_window, incoming)

            result[i] = np.mean(sorted_window[:n_lowest])

        return result

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
        has_transients = params.get("has_transients", self.exp_params["has_transients"])
        if not has_transients or len(self._peak_idxes) == 0:
            return
        ISI_factor = params.get("ISI_factor", self.pd_params["ISI_factor"])

        peak_times = self.peak_times
        if len(peak_times) < 2:
            print("Cannot remove transients, not enough peaks.")
            return

        avg_ISI = np.average(peak_times[1:] - peak_times[:-1])
        if (peak_times[1] - peak_times[0]) > ISI_factor * avg_ISI:
            self._peak_idxes = np.array(self._peak_idxes[1:])

    def apply_low_threshold(self, params):
        """Removes all peaks below a percentage of the max amplitude."""
        if len(self.peak_amplitudes) == 0:
            return
        threshold = params.get("low_amp_threshold", self.pd_params["low_amp_threshold"])

        max_peak = max(self.peak_amplitudes)
        self._peak_idxes = np.array(
            [p for p in self._peak_idxes if self.dff[p] > threshold * max_peak]
        )

    def reconcile_manual_peaks(self, params):
        """Reconcile the calculated peaks with manually corrected data.

        The manual data is written through the GUI."""
        corrected_peaks = params.get(
            "corrected_peaks", self.config.get_corrected_peaks(self.name)
        )

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

    def update_dsna_start(self, params):
        if not self.exp_params.get("has_dsna", False):
            return
        self.dsna_start = self.get_dsna_start(params["freq"])

    def detect_peaks(self, freq=0.0025):
        self._peak_idxes, filtered_dff = self.calculate_peaks(freq_cutoff=freq)
        self.filtered_dff = filtered_dff
        self.dsna_start = self.get_dsna_start(freq)

        stages = [
            (self.remove_transients, {}),
            (self.apply_low_threshold, {}),
            (self.reconcile_manual_peaks, {}),
            (self.update_dsna_start, {"freq": freq}),
        ]

        self.process_peaks(stages)

    def detect_oscillations(self, after_ep=2, offset=10):
        # oscillations only happen in phase 2, which should be at least after ep 2:
        if len(self.peak_idxes) <= after_ep:
            return None

        mask = np.zeros_like(self.dff, dtype=np.bool_)
        peak_bounds = self.peak_bounds_indices
        for s, e in peak_bounds:
            mask[s : e + offset] = 1
        p2_start = self.peak_idxes[after_ep]
        dff = self.dff.copy()
        dff[mask] = 0
        dff = dff[p2_start : self.trim_idx]
        order0_idxes = spsig.find_peaks(dff, height=0.1, distance=5, prominence=0.07)[0]

        return order0_idxes + p2_start

    def get_filtered_signal(self, freq_cutoff, low_pass=True):
        """Applies a low or high pass filter and returns the inverse result."""
        N = len(self.dff[: self.trim_idx])
        freqs = np.fft.rfftfreq(N, 1 / self.fs)
        fft = np.fft.rfft(self.dff[: self.trim_idx])
        if low_pass:
            mask = freqs < freq_cutoff
        else:
            mask = freqs > freq_cutoff
        filtered_fft = fft * mask
        filtered_signal = np.fft.irfft(filtered_fft, n=N)

        return filtered_signal

    def filter_peaks_by_local_context(
        self, signal, peak_indices, window_size=300, value=75, method="percentile"
    ):
        """Filter pre-detected peaks by comparing their height to a local threshold.

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

    def port_peaks(self, peaks, target_signal, search_window=30, peak_height_thres=70):
        """Changes peaks index to the highest peak amplitude on a target signal.

        Parameters:
            peaks (list[int]):
                list of peak indices.
            target_signal (list):
                Signal that will be used to determine final peak position.
            search_window (int):
                Half the size of the window used to look for the peak in the target signal.
            peak_height_thres (int):
                Should be within 0 ~ 100. Minimum percentage necessary for the
                ported peak relative to the local maximum.
        """
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
                if ph >= (peak_height_thres / 100) * max_peak
            )
            local_peak_indices.append(left + leftmost_peak)

        return np.array(local_peak_indices)

    def calculate_peaks(self, freq_cutoff):
        """Detect peaks based on the low-passed iFFT signal.

        Peaks found in the freq domain are used as anchor points in the actual
        signal to determine peak indices.
        """
        filtered_dff = self.get_filtered_signal(freq_cutoff)

        fft_height = self.pd_params["fft_height"]
        fft_prominence = self.pd_params["fft_prominence"]
        peak_indices, _ = spsig.find_peaks(
            filtered_dff, height=fft_height, prominence=fft_prominence
        )

        pp_ws = self.pd_params["port_peaks_window_size"]
        pp_thres = self.pd_params["port_peaks_thres"]
        local_peak_indices = self.port_peaks(
            peak_indices,
            self.dff[: self.trim_idx],
            search_window=pp_ws,
            peak_height_thres=pp_thres,
        )

        local_ws = self.pd_params["local_thres_window_size"]
        local_value = self.pd_params["local_thres_value"]
        local_method = self.pd_params["local_thres_method"]
        peaks = self.filter_peaks_by_local_context(
            self.dff,
            local_peak_indices,
            window_size=local_ws,
            value=local_value,
            method=local_method,
        )

        return np.array(peaks), filtered_dff

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
        """Computes the z score for each Savitzky-Golay-filtered sample, and
        removes all points after reaching `trim_zscore`."""
        tomato_savgol = spsig.savgol_filter(self.struct, 21, 2, deriv=0)
        zscored_tomato = zscore(tomato_savgol)
        zscored_tomato -= self.compute_baseline(zscored_tomato, window_size=10)

        trim_points = np.where(np.abs(zscored_tomato) > trim_zscore)[0]
        if len(trim_points) == 0:
            trim_idx = len(self.time)
        else:
            trim_idx = trim_points[0]
        return trim_idx

    def compute_peak_bounds(self, rel_height=0.97, peak_idxes=None):
        """Computes properties of each dff peak using spsig.peak_widths."""
        if peak_idxes is None:
            peak_idxes = self.peak_idxes
        if len(peak_idxes) == 0:
            return np.array([])
        dff = self.dff[: self.trim_idx].copy()
        # make sure that the peak we pass is not bound by local points
        dff[peak_idxes] += 3
        _, _, start_idxs, end_idxs = spsig.peak_widths(
            dff, peak_idxes, rel_height, wlen=150
        )

        start_idxs = start_idxs.astype(np.int64)
        end_idxs = end_idxs.astype(np.int64)
        peak_bounds_indices = np.vstack((start_idxs, end_idxs)).T

        corrected_peaks = self.config.get_corrected_peaks(self.name)

        if not corrected_peaks:
            return peak_bounds_indices

        # reconcile the peak_bounds with manually changed peak widths here:
        # if a peak has associated manual width data, overwrite that peak's bounds
        peaks_to_bounds = {
            p: [s, e] for (p, s, e) in zip(peak_idxes, start_idxs, end_idxs)
        }

        manual_peak_bounds = corrected_peaks.get("manual_widths", None)

        if manual_peak_bounds:
            # must convert the data from json file to np type:
            manual_peak_bounds = {
                np.int64(k): [np.int64(n) for n in v]
                for k, v in manual_peak_bounds.items()
            }

            for peak in manual_peak_bounds:
                peaks_to_bounds[peak] = manual_peak_bounds[peak]

        return np.array(list(peaks_to_bounds.values()))

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
        """Returns local peaks that happen within an episode."""
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
        if self.peak_idxes.size == 0:
            return (0, 0)
        first_peak = self.peak_idxes[0]
        dff = self.dff[: self.trim_idx]

        lp_before, _ = spsig.find_peaks(
            dff[first_peak:split_idx], height=height, prominence=prominence
        )
        lp_after, _ = spsig.find_peaks(
            dff[split_idx:], height=height, prominence=prominence
        )

        return (len(lp_before), len(lp_after))

    def calculate_STFT(self, dff, fs=1 / 6, fft_size=600, noverlap=450):
        """Calculate the Short Time Fourier Transform for a dff signal.

        It replaces None values with 1e-11.

        Parameters:
            dff (arr):
                Preprocessed dff.
            fs (float):
                Sampling rate.
            fft_size (int):
                Num frames in each segment.
            noverlap (int):
                Num frames to overlap between segments.

        Returns:
            f (arr):
                STFT frequency bins.
            t (arr):
                STFT time columns.
            magnitude (array):
                STFT magnitude (excludes phase).
        """
        dff = np.where(dff == None, 1e-11, dff).astype(float)
        f, t, Zxx = spsig.stft(dff, fs, nperseg=fft_size, noverlap=noverlap)

        return f, t, Zxx

    def apply_hipass_filter(self, time, dff, cutoff, fs=1 / 6, numtaps=501):
        """Apply a high pass finite impulse response filter to the dff signal.

        Parameters:
            time (arr)
                Timepoints associated to each dff point.
            dff (arr)
                dff values.
            fs (float):
                Sampling rate.
            fft_size (int):
                Num frames in each segment.
            numtaps (int):
                Num of coefficients in the filter.

        Returns:
            hipass_time (arr):
                input time with filter applied.
            hipass_dff (arr):
                input dff with filter applied.
        """
        delay = int((numtaps - 1) / 2)
        padded_dff = np.pad(dff, (0, delay), mode="constant")
        fir_hipass = spsig.firwin(
            numtaps, cutoff=cutoff, fs=fs, pass_zero=False
        )  # filter coeffs
        hipass_dff = spsig.lfilter(fir_hipass, [1.0], padded_dff)[delay:-delay]
        hipass_time = time[: len(hipass_dff)]
        return hipass_time, hipass_dff

    def apply_lopass_filter(self, time, dff, cutoff, fs=1 / 6, numtaps=501):
        """Apply a low pass finite impulse response filter to the dff signal.

        Parameters:
            time (arr)
                Timepoints associated to each dff point.
            dff (arr)
                dff values.
            fs (float):
                Sampling rate.
            fft_size (int):
                Num frames in each segment.
            numtaps (int):
                Num of coefficients in the filter

        Returns:
            lopass_time (arr):
                input time with filter applied
            lopass_dff (arr):
                input dff with filter applied
        """
        delay = int((numtaps - 1) / 2)
        padded_dff = np.pad(dff, (0, delay), mode="constant")
        fir_hipass = spsig.firwin(
            numtaps, cutoff=cutoff, fs=fs, pass_zero=True
        )  # filter coeffs
        lopass_dff = spsig.lfilter(fir_hipass, [1.0], padded_dff)[delay:-delay]
        lopass_time = time[: len(lopass_dff)]
        return lopass_time, lopass_dff

    def get_dsna_start(self, freq):
        if not self.exp_params.get("has_dsna", False):
            return None

        manual_dsna = self.config.get_corrected_dsna_start(self.name)

        if manual_dsna is not None and manual_dsna >= 0:
            return manual_dsna

        tp = TracePhases(self)
        dsna_start = tp.get_dsna_start(freq)
        if dsna_start == -1:
            return self.trim_idx

        return dsna_start
