import bisect
from enum import Enum

import numpy as np
import scipy.signal as spsig
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import percentile_filter
from scipy.integrate import trapezoid
from scipy.stats import zscore
from scipy.ndimage import label

from snazzy_analysis import Config, FrequencyAnalysis


class BaselineStrategies(Enum):
    """Enumerates the methods used to calculate baseline."""

    BASELINE = "baseline"
    LOCAL_MINIMA = "local_minima"


class Trace:
    """Calculates dff and peak data for the resulting trace."""

    def __init__(
        self,
        name,
        activity,
        config: Config,
    ):
        self.name = name
        self.time = activity[:, 0]
        self.active = activity[:, 1]
        self.struct = activity[:, 2]
        self.config = config
        self.pd_params = config.get_pd_params()
        self.exp_params = config.get_exp_params()

        self.trim_idx = self.get_trim_index()
        self.dff = self.compute_dff()

        # list of peaks that were manually added / removed or
        # automatically transfered to valid event
        self.to_add = []
        self.to_remove = []
        self.to_transfer = []

        self._event_idxes = None
        self._peak_idxes = None
        self._mini_idxes = None
        self._peak_bounds_indices = None
        self.filtered_dff = None

        
        self._peak_proms = None
        self._mini_proms = None
        # these bounds are calculated with the prominences above
        self._mini_bounds_indices = None

    @property
    def event_idxes(self):
        if self._event_idxes is None:
            self.detect_events()
        return self._event_idxes

    @event_idxes.setter
    def event_idxes(self, event_idxes):
        self._event_idxes = event_idxes

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
    def peak_proms(self):
        if self._peak_proms is None:
            self.assign_prominences(min_base=0.01)
        return self._peak_proms

    @peak_proms.setter
    def peak_proms(self, peak_proms):
        self._peak_proms = peak_proms

    @property
    def peak_times(self):
        if len(self.peak_idxes) == 0:
            return []
        return self.time[self.peak_idxes]

    @property
    def mini_idxes(self):
        if self._mini_idxes is None:
            self.detect_minis()
        return self._mini_idxes

    @mini_idxes.setter
    def mini_idxes(self, mini_idxes):
        self._mini_idxs = mini_idxes

    @property
    def mini_proms(self):
        if self._mini_proms is None:
            self.assign_prominences(min_base=0.01)
        return self._mini_proms

    @mini_proms.setter
    def mini_proms(self, mini_proms):
        self._mini_proms = mini_proms

    @property
    def mini_bounds_indices(self):
        if self._mini_bounds_indices is None:
            self.set_mini_widths()
        return self._mini_bounds_indices

    @mini_bounds_indices.setter
    def mini_bounds_indices(self, mini_bounds_indices):
        self._mini_bounds_indices = mini_bounds_indices

    @property
    def mini_times(self):
        if len(self.mini_idxes) == 0:
            return []
        return self.time[self.mini_idxes]

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
        dff_strategy_name = self.pd_params.get("dff_strategy", "")

        dff_strategy = BaselineStrategies(dff_strategy_name)
        if dff_strategy == BaselineStrategies.BASELINE:
            baseline = self.compute_baseline(ratiom_signal, window_size)
        elif dff_strategy == BaselineStrategies.LOCAL_MINIMA:
            baseline = self.average_n_lowest_window(
                ratiom_signal, window_size, n_lowest=11
            )
        return (ratiom_signal - baseline) / baseline

    def preprocess_dff(self, duration=5000, onset_pad=300):
        """Adjust dff to a given duration.

        If embryo hatches, only data from onset to hatching are included.
        If activity after onset does not continue for the entire duration,
        the dff is padded.
        The resulting time and dff have the same length.

        Parameters:
            duration (int):
                Target num of frames for time and dff.
            onset_pad (int):
                Num of frames to include before onset.

        Returns:
            time_processed (arr):
                Trimmed/extended time (mins).
            dff_processed (arr):
                Trimmed/extended dff.
        """

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

        if duration > len(dff):
            pad_size = duration - len(dff)
            dff_processed = np.array(list(dff) + [0] * pad_size, dtype=object)
        else:
            dff_processed = dff[0:duration]

        acq_period = self.exp_params["acquisition_period"]
        # acquisition period is in seconds, but time_processed should be mins
        final_timepoint = duration * acq_period / 60
        time_processed = np.arange(0, final_timepoint, acq_period / 60)

        return time_processed, dff_processed, start_index

    def align_time_to_onset(self, time_in_sec, onset_pad=300):
        time_in_sec = np.array(time_in_sec)
        # find new onset
        if self.peak_bounds_indices is not None and len(self.peak_bounds_indices) > 0:
            onset = self.peak_bounds_indices[0][0]
        else:
            onset = 0
        offset = self.time[onset]
        return time_in_sec - offset

    def normalize_signal(self, signal, baseline_amplitude, step=50):
        """Normalize signal to range 0..1, set values to zero where below baseline amplitude."""
        out = np.zeros_like(signal, dtype=float)

        for start in range(0, len(signal), step):
            end = np.min([start + step, len(signal)])
            s = signal[start:end]

            s_max = np.max(s)
            s_min = np.min(s)

            if np.median(s) > baseline_amplitude:
                s_norm = (s - s_min) / (s_max - s_min)
            else:
                s_norm = 0

            out[start:end] += s_norm

        return out

    def estimate_baseline_amplitude(self, wlen=150):
        """
        Estimate the baseline amplitude from the wlen samples immediately
        preceding the first episode.

        If fewer than wlen samples are available, all samples preceding the
        first episode are used.
        """
        first_episode_start = int(self.peak_bounds_indices[0][0])
        start = max(0, first_episode_start - wlen)
        stop = min(first_episode_start, len(self.dff))

        presna = self.dff[start:stop]

        return np.median(presna)

    def find_localpeaks_norm_signal(self, norm_signal, parameters):
        """
        Find peaks in a normalized signal.

        Parameters
        ----------
        norm_signal : array-like
            Normalized signal.
        parameters : dict
            Keyword arguments passed to scipy.signal.find_peaks. For example::

                {
                    "height": 0,
                    "prominence": 0.05,
                    "distance": 1,
                    "width": (0, None),
                    "wlen": 150,
                    "rel_height": 0.9,
                }

        Returns
        -------
        numpy.ndarray
            Peak indices relative to norm_signal.
        """
        if parameters is None:
            parameters = {
                "height": 0.2,
                "prominence": 0.2,
                "distance": 1,
                "width": (0, None),
                "wlen": 40,
                "rel_height": 0.9,
            }

        peak_indices, _ = spsig.find_peaks(norm_signal, **parameters)

        return peak_indices.astype(int)

    def compute_local_peaks(self, prominence_ratio, height=0.1, wlen=30):
        """Returns local peaks that happen within an episode."""
        starts = [x[0] for x in self.peak_bounds_indices]
        pairs = list(zip(starts[:-1], starts[1:]))
        # include last ep boundaries
        pairs.append(self.peak_bounds_indices[-1])
        local_peaks = []
        local_peak_props = []

        for pi, (s, e) in zip(self.peak_idxes, pairs):
            if e >= self.trim_idx:
                break
            if pi == e:
                continue
            local_trace = self.dff[s:e]
            peak_indices, props = spsig.find_peaks(
                local_trace,
                prominence=self.dff[pi] * prominence_ratio,
                height=height,
                wlen=wlen,
            )
            local_peaks.extend(peak_indices + s)
            local_peak_props.extend(props)

        return np.array(local_peaks), local_peak_props

    def find_localpeaks(
        self,
        min_amplitude=0.08,
        exclude_radius=10,
        prominence_ratio=0.3,
        wlen=30,
        baseline_wlen=150,
        post_sna_step=50,
        normalized_peak_params=None,
    ):
        """Detect local peaks using peak finding params scaled to the current segment."""

        dff = np.asarray(self.dff[: self.trim_idx], dtype=float).copy()

        local_peak_idxs, local_peak_props = self.compute_local_peaks(
            prominence_ratio,
            height=min_amplitude,
            wlen=wlen,
        )

        # Because of overlapping windows, there could be duplicates
        local_peak_idxs = np.unique(local_peak_idxs)

        keep = np.ones(len(local_peak_idxs), dtype=bool)

        # Keep only peaks after the first episode/burst
        first_onset = self.peak_bounds_indices[0][0]
        keep &= local_peak_idxs > first_onset

        close_to_ep_peak = np.any(
            np.abs(local_peak_idxs[:, None] - self.peak_idxes[None, :])
            <= exclude_radius,
            axis=1,
        )

        keep &= ~close_to_ep_peak

        # Optional amplitude criterion in original dF/F units
        keep &= dff[local_peak_idxs] >= min_amplitude

        # return all_local_peak_idxs[keep]
        local_peak_idxs = local_peak_idxs[keep]
        local_peak_props = [
            lpp for lpp, should_keep in zip(local_peak_props, keep) if should_keep
        ]

        # add local peaks found after sna here (after peak_bounds_indices[-1][1])
        last_peak_end = int(self.peak_bounds_indices[-1][1])

        if last_peak_end < self.trim_idx:
            baseline_amplitude = self.estimate_baseline_amplitude(wlen=baseline_wlen)

            post_trace = dff[last_peak_end:]

            post_trace_norm = self.normalize_signal(
                post_trace, baseline_amplitude=baseline_amplitude, step=post_sna_step
            )

            if normalized_peak_params is None:
                normalized_peak_parameters = {
                    "height": 0.6,
                    "prominence": 0.4,
                    "distance": 1,
                    "width": (0, None),
                    "wlen": min(30, len(post_trace_norm)),
                    "rel_height": 0.9,
                }

            post_peak_idxs = self.find_localpeaks_norm_signal(
                post_trace_norm,
                normalized_peak_parameters,
            )
            # Convert indices from post-SNA coordinates to dff coordinates.
            post_peak_idxs = post_peak_idxs + last_peak_end
            post_peak_amps = dff[post_peak_idxs]

            # Ignore peak min height check if not enough peaks found
            if len(self.peak_idxes) < 2:
                avg_last_two_peak_amps = 0
            else:
                avg_last_two_peak_amps = (
                    dff[self.peak_idxes[-2]] + dff[self.peak_idxes[-1]]
                ) / 2

            keep = post_peak_amps > 0.75 * avg_last_two_peak_amps
            post_peak_idxs = post_peak_idxs[keep]
        else:
            post_peak_idxs = np.array([], dtype=int)

        all_local_peak_idxs = np.concatenate(
            (
                np.asarray(local_peak_idxs, dtype=int),
                np.asarray(post_peak_idxs, dtype=int),
            )
        )

        return all_local_peak_idxs, local_peak_props

    def get_bursts_only(self):
        time = self.time
        dff = self.dff
        burst_bounds = self.peak_bounds_times
        burst_indices = [
            (
                np.searchsorted(time, start, side="left"),
                np.searchsorted(time, stop, side="right"),
            )
            for start, stop in burst_bounds
        ]
        dff = np.array(dff)

        mask = np.zeros(len(dff), dtype=bool)
        for start, stop in burst_indices:
            mask[start:stop] = True

        return dff[mask]

    def compute_ratiom_gcamp(self):
        """Computes the ratiometric GCaMP signal by dividing the raw GCaMP
        signal by the savgol filtered tdTomato signal."""
        smoothed = spsig.savgol_filter(
            self.struct[: self.trim_idx], 151, 2, deriv=0
        )
        return self.active[: self.trim_idx] / smoothed

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
            print(
                f"WARN: window_size should be an odd number. Using ws={window_size-1}"
            )
            window_size -= 1
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

        This assumes that bouts are sparse. To handle edges, both edges are reflected.
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

    def remove_below_threshold(self, params):
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
            to_add = corrected_peaks.get("manual_peaks", [])
            to_remove = corrected_peaks.get("manual_remove", [])
            wlen = corrected_peaks.get("wlen", 2)

            filtered_peaks = [
                p
                for p in self._peak_idxes
                if not any(abs(p - rp) < wlen for rp in to_remove)
            ]

            filtered_peaks = np.array(
                sorted(list(set(filtered_peaks + to_add))), dtype=np.int64
            )

            valid_peaks = self.get_valid_peaks(filtered_peaks) # moves peak_idxs to valid peak in events
            to_transfer = filtered_peaks[filtered_peaks != valid_peaks]

            self.to_add = to_add
            self.to_remove = to_remove
            self.to_transfer = to_transfer
            self._peak_idxes = valid_peaks

    def get_valid_peaks(self, peak_idxs, max_dist=20):
        dff = self.dff.copy()
        dff = dff[:self.trim_idx]
        event_idxs = self.event_idxes

        valid_peaks = np.array([
            (print("port"), event_idxs[np.argmax(np.where(np.abs(event_idxs - p) <= max_dist, dff[event_idxs], -np.inf))], p)[1] # port burst idx to closest peak
            if np.min(np.abs(event_idxs - p)) > 0  # burst is not a valid peak
            and np.min(np.abs(event_idxs - p)) < 20 # but burst is close to a peak
            and  np.any(dff[max(0, p-5):p] > dff[p]) # and burst is less than 5 prev values
            else p
            for p in peak_idxs
        ])
        if len(peak_idxs) != len(valid_peaks):
            raise ValueError("peak_idxs and valid_peaks must have the same length")
        return valid_peaks

    def detect_events(self, height=0.05, prominence=0.08):
        dff = self.dff.copy()
        dff = dff[:self.trim_idx]
        event_idxs, _ = spsig.find_peaks(dff, height=height, prominence=prominence)
        self._event_idxes = event_idxs

    def detect_peaks(self, freq=0.0025):
        self._peak_idxes, filtered_dff = self.calculate_peaks(freq_cutoff=freq)
        self.filtered_dff = filtered_dff

        stages = [
            (self.remove_transients, {}),
            (self.remove_below_threshold, {}),
            (self.reconcile_manual_peaks, {}),
        ]

        self.process_peaks(stages)

    def detect_localpeaks(self):
        self._localpeak_idxes, self._localpeak_props = self.find_localpeaks()

    def filter_peaks_by_local_threshold(
        self, signal, peak_indices, window_size=300, value=75
    ):
        """Filter pre-detected peaks by comparing their height to a local threshold.

        Peaks that are not within a percentile of the values in the window are filtered out.

        Parameters:
            signal (np.ndarray): Original signal.
            peak_indices (np.ndarray): Indices of peaks (e.g., from find_peaks).
            window_size (int): Size of the window to determine local threshold.
            value (int): Factor to determine local threshold.

        Returns:
            filtered_peaks (np.ndarray): Indices of peaks that passed local thresholding.
        """
        filtered_peaks = []

        for i in peak_indices:
            if signal[i] > self.local_threshold(signal, i, window_size, value):
                filtered_peaks.append(i)

        return np.array(sorted(filtered_peaks))

    def local_threshold(self, signal, idx, window_size, value):
        half_win = window_size // 2

        start = max(0, idx - half_win)
        end = min(len(signal), idx + half_win)
        window = signal[start:end]

        local_thresh = np.percentile(np.abs(window), value)

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
            if not any(local_peak_heights):
                continue
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
        filtered_dff = FrequencyAnalysis.apply_lopass_filter(
            self.dff[: self.trim_idx], freq_cutoff
        )

        fft_height = self.pd_params["fft_height"]
        fft_prominence = self.pd_params["fft_prominence"]
        peak_indices, _ = spsig.find_peaks(
            filtered_dff, height=fft_height, prominence=fft_prominence
        )

        pp_ws = self.pd_params["port_peaks_window_size"]
        pp_thres = self.pd_params["port_peaks_thres"]
        dff_peak_indices = self.port_peaks(
            peak_indices,
            self.dff[: self.trim_idx],
            search_window=pp_ws,
            peak_height_thres=pp_thres,
        )

        return np.array(dff_peak_indices), filtered_dff

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
            trim_idx = len(self.time) - 1
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
            [np.trapezoid(pslice, tslice) for pslice, tslice in peak_time_slices]
        )
        return peak_aucs

    def count_local_peaks(self, height=0.02, prominence=0.01):
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

    def calculate_activity_density_ratio(self):
        time = self.time
        dff = self.dff
        burst_bounds = self.peak_bounds_indices

        burst_densities = []
        interval_densities = []
        for i in range(len(burst_bounds) - 1):  # excludes last burst
            start_bound, end_bound = burst_bounds[i][0], burst_bounds[i][1]
            next_start_bound = burst_bounds[i + 1][0]

            burst = slice(start_bound, end_bound + 1)
            interval = slice(end_bound, next_start_bound)

            burst_duration = time[end_bound + 1] - time[start_bound]
            interval_duration = time[next_start_bound] - time[end_bound]

            # dff x burst
            burst_auc = trapezoid(dff[burst], time[burst])
            burst_density = burst_auc / burst_duration

            # dff x interval
            interval_auc = trapezoid(dff[interval], time[interval])
            interval_density = interval_auc / interval_duration

            burst_densities.append(burst_density)
            interval_densities.append(interval_density)

        burst_densities = np.array(burst_densities)
        interval_densities = np.array(interval_densities)
        return burst_densities, interval_densities

    def detect_minis(self, peak_win=5):
        """Minis are all events that are not a burst."""
        mini_idxes = [
        e for e in self.event_idxes
        if (self.peak_idxes[0] < e # must be after first burst
            and not np.any(np.abs(self.peak_idxes - e) <= peak_win)) # and peak_win samples away from burst idx
        ]
        self._mini_idxes = mini_idxes

    def calculate_left_prominences(self, dff, events, min_base=0.01):
        _, left_bases, right_bases = spsig.peak_prominences(dff, events)
        left_prominences = [dff[e] - np.max([dff[lb], min_base]) for e, lb in zip(events, left_bases)]
        return np.array(left_prominences), np.array(left_bases), np.array(right_bases)

    def calculate_regular_prominences(self, dff, events, min_base=0.01):
        _, left_bases, right_bases = spsig.peak_prominences(dff, events)
        new_prom = dff[events] - np.maximum(
            np.maximum(dff[left_bases], dff[right_bases]),
            min_base
        )
        return np.array(new_prom), np.array(left_bases), np.array(right_bases)

    def assign_prominences(self, min_base=0.01):
        dff = self.dff.copy()
        dff = dff[:self.trim_idx]
        peak_idxes = self.peak_idxes
        mini_idxes = self.mini_idxes

        peak_proms, peak_left, peak_right = self.calculate_left_prominences(dff, peak_idxes, min_base=min_base)
        mini_proms, mini_left, mini_right = self.calculate_regular_prominences(dff, mini_idxes, min_base=min_base)

        peaks = np.concatenate([peak_idxes, mini_idxes])
        prom = np.concatenate([peak_proms, mini_proms])
        lb = np.concatenate([peak_left, mini_left])
        rb = np.concatenate([peak_right, mini_right])

        order = np.argsort(peaks)
        peaks, prom, rb = (arr[order] for arr in (peaks, prom, rb))

        new_prom = prom.copy()
        bases = dff[peaks] - prom

        for i in range(1, len(peaks)):
            for j in range(i):
                same_base = np.isclose(bases[i], bases[j], atol=0.001, rtol=0)
                no_baseline = np.all(dff[peaks[j]:peaks[i]] >= min_base)

                if same_base and no_baseline:
                    if rb[i-1] < peaks[i]:
                        new_prom[i] = dff[peaks[i]] - max(
                            dff[rb[i-1]], dff[lb[i]], min_base
                        )
                    break

        inverse = np.argsort(order)
        new_prom_peak, new_prom_mini = np.split(new_prom[inverse], [len(peak_idxes)])

        changed = ~np.isclose(prom, new_prom)
        changed_prom_peaks = peaks[changed]
        # for i in np.where(~np.isclose(prom, new_prom))[0]:
        #     print(f"Peak {time[peaks[i]]}: {prom[i]:.3f} → {new_prom[i]:.3f}")

        self._peak_proms = new_prom_peak
        self._mini_proms = new_prom_mini

    def calculate_boundless_widths(self, dff, events, prominences, rel_height=0.98, floor=0.01):
        left_limits = [([i for i in range(e) if dff[i] <= floor] or [0])[-1] for e in events]
        left_limits = np.array(left_limits)
        right_limits = [([i for i in range(e+1, len(dff)) if dff[i] <= floor] or [len(dff)-1])[0] for e in events]
        right_limits = np.array(right_limits)

        _, _, left_ips, right_ips = spsig.peak_widths(
            dff, events, rel_height=rel_height, prominence_data=(prominences, left_limits, right_limits)
        )
        widths = np.round(right_ips - left_ips).astype(int)
        starts = [np.floor(li).astype(int) for li in left_ips]
        ends = [np.ceil(si).astype(int) for si in right_ips]
        return widths, starts, ends, left_limits, right_limits

    def set_mini_widths(self):
        dff = self.dff.copy()
        dff = dff[:self.trim_idx]
        print(self.peak_idxes)

        _, peak_starts, peak_ends, _, _ = self.calculate_boundless_widths(dff, self.peak_idxes, self.peak_proms)
        _, mini_starts, mini_ends, _, _  = self.calculate_boundless_widths(dff, self.mini_idxes, self.mini_proms)

        # the peak starts/ends could be used to replace current bounds
        bounds_indices = [(s, e) for s,e in zip(mini_starts, mini_ends)]
        self._mini_bounds_indices = bounds_indices
