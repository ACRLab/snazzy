from typing import Any

import numpy as np

from snazzy_analysis import Trace


def local_peak_at(x, signal, wlen):
    local_max_index = np.argmax(signal)
    adjusted_max = local_max_index + x - wlen
    return int(adjusted_max)


class PeakMatcher:
    def add_peak(self, x: int, trace: Trace, manual_remove: list[int], wlen: int = 2):
        """Add a new peak in the vicinity of `x`.

        The new peak is calculated as the local maximum of a window centered at `x`,
        that ranges `wlen` in each direction.

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
        new_peak = local_peak_at(x, trace.dff[window], wlen)
        if new_peak in trace.peak_idxes:
            return new_peak, trace.peak_idxes, None
        new_arr = np.append(trace.peak_idxes, new_peak)
        new_arr.sort()

        # if we add a peak in a place where there's a manually removed peak,
        # that manually removed entry must be erased:
        to_remove = None
        if manual_remove:
            if new_peak in manual_remove:
                i = manual_remove.index(new_peak)
                to_remove = manual_remove[:i] + manual_remove[i + 1 :]

        return new_peak, new_arr, to_remove

    def remove_peak(
        self,
        x: int,
        trace: Trace,
        manual_add: list[int],
        manual_widths: dict[str, Any],
        wlen=2,
    ):
        """Remove all peaks that are within ±`wlen` of `x`.

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
            filtered_peak_widths (int):
                Updated manual widhts dict without the peaks that were removed.
        """
        target = (trace.peak_idxes >= x - wlen) & (trace.peak_idxes <= x + wlen)
        removed = trace.peak_idxes[target].tolist()
        new_arr = trace.peak_idxes[~target]

        # if we remove a peak where there's a manually added peak,
        # that manually added peak entry must be erased
        to_add = None
        if manual_add:
            filtered_manual_add = [
                p for p in manual_add if p < x - wlen or p > x + wlen
            ]
            to_add = filtered_manual_add

        filtered_peak_widths = None
        if manual_widths:
            filtered_peak_widths = {
                p: data
                for p, data in manual_widths.items()
                if int(p) < x - wlen or int(p) > x + wlen
            }

        return removed, new_arr, to_add, filtered_peak_widths
