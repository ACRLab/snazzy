# Peak Detection

Peak detection is the core feature of `pasna_analysis`.
From the detected peaks, we derive most of the metrics used in this package: peak widths, amplitudes, rise times, decay times, and more.
The algorithm consists of several steps, each with parameters that can be fine-tuned for optimal detection.
To understand how to adjust these parameters effectively, it's important to first understand the entire peak detection algorithm.

### 1. Peak Detection on a Low-Pass Filtered Signal

The ΔF/F (dff) trace is filtered in the frequency domain using a `freq_cutoff` parameter: all frequencies above this value are removed, and the remaining low-frequency components are used to reconstruct the filtered trace.
This acts as a smoothing step, reducing oscillations and very short-duration peaks that do not correspond to actual activity bursts.

The `freq_cutoff` can be adjusted in the GUI using a slider, and the reconstructed signal is updated in real time.
The default value of `0.025` works well for many traces, but traces with high-frequency noise may require a lower cutoff.

Once we have the filtered trace, peaks are detected using the parameters `fft_height` and `fft_prominence`.
The `fft_height` parameter is especially important because the reconstructed signal often contains minor ripples before the first real burst.
These are easy to identify, as they usually do not correspond to peaks in the original ΔF/F trace.
`fft_prominence` complements `fft_height` by measuring how much a point stands out from its surrounding baseline in order to be marked as a peak.

### 2. Align Peaks in the Original Signal

After detecting peaks in the filtered signal, the peak indices are mapped back to the original ΔF/F trace.
This step is necessary because frequency-domain transformations can slightly shift peak positions.

Next, the detected peaks are validated against a local threshold.
For example, if a peak is misidentified between bursts, it will likely be discarded due to the presence of higher values nearby.

### 3. Optional Post-Processing

Some post-processing operations can improve peak detection for specific types of traces.

Certain traces may exhibit a large burst at the beginning of the imaging session: an artifact that should be removed.
In such cases, the `remove_transients` function can be applied.
It detects and removes initial bursts if their interval is significantly longer than the average of subsequent bursts.

Another post-processing step removes low-amplitude peaks that are likely false positives.
Peaks below a specified percentage of the maximum peak amplitude are discarded.

Finally, in the GUI, users can manually add or remove peaks.
These manual edits are used to update the set of calculated peaks.
