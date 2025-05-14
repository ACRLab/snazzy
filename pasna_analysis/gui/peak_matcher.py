from typing import Any

import numpy as np

from pasna_analysis import Trace


def local_peak_at(x, signal, wlen):
    local_max_index = np.argmax(signal)
    adjusted_max = local_max_index + x - wlen
    return int(adjusted_max)


class PeakMatcher:
    def add_peak(self, x: int, trace: Trace, manual_remove: list[int], wlen: int = 10):
        """Adds a new peak in the vicinity of `x`.

        The new peak is calculated as the local maximum near `x`, in a window of 2*wlen points.

        Parameters:
            x (int):
                Center of the window to mark the peak.
            trace (Trace):
                Trace signal used to determine the local maximum.
            manual_remove (list[int]):
                List of indices manually marked to removal.
            wlen (int):
                Window length to look for local maximum

        Returns:
            new_peak (int):
                Index of the new peak
            new_peaks (list[int]):
                Updated list with all peaks
            to_remove (list[int]):
                Updated list of manually removed peaks
        """
        window = slice(x - wlen, x + wlen)
        new_peak = local_peak_at(x, trace.order_zero_savgol[window], wlen)
        new_arr = np.append(trace.peak_idxes, new_peak)
        new_arr.sort()

        # if we add a peak in a place where there's a manually removed peak,
        # that manually removed entry must be erased:
        to_remove = None
        if manual_remove:
            try:
                peak = next(p for p in manual_remove if x - wlen <= p <= x + wlen)
                i = manual_remove.index(peak)
                to_remove = manual_remove[:i] + manual_remove[i + 1 :]
            except StopIteration:
                pass

        return new_peak, new_arr, to_remove

    def remove_peak(
        self,
        x: int,
        trace: Trace,
        manual_add: list[int],
        manual_widths: dict[str, Any],
        wlen=10,
    ):
        """Removes peaks that are within `wlen` of `x`.

        Parameters:
            x (int):
                Center of the window to mark the peak.
            trace (Trace):
                Trace signal used to determine the local maximum.
            manual_add (list[int]):
                List of manually added peak indices.
            manual_widths (dict):
                List of mannually added width boundaries.
            wlen (int):
                Window length to look for local maximum.

        Returns:
            removed (list[int]):
                List of peaks to be removed.
            new_arr (list[int]):
                Updated list with all peaks.
            to_add (list[int]):
                Updated list of manually added peaks.
            peak_width_to_remove (int):
                Key that should be removed from manual widths dict.
        """
        target = (trace.peak_idxes >= x - wlen) & (trace.peak_idxes <= x + wlen)
        removed = trace.peak_idxes[target].tolist()
        new_arr = trace.peak_idxes[~target]

        # if we remove a peak where there's a manually added peak,
        # that manually added peak entry must be erased
        to_add = None
        if manual_add:
            try:
                # FIXME: if you remove more than one element, this update method fails
                peak = next(p for p in manual_add if x - wlen <= p <= x + wlen)
                i = manual_add.index(peak)
                to_add = manual_add[:i] + manual_add[i + 1 :]
            except StopIteration:
                pass
        peak_width_to_remove = None
        if manual_widths:
            try:
                peak_width_to_remove = next(
                    int(p) for p in manual_widths if x - wlen <= int(p) <= x + wlen
                )
            except StopIteration:
                pass

        return removed, new_arr, to_add, peak_width_to_remove
