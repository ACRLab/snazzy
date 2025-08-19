# Hathching point calculation

Hatching happens when the fruit-fly embryo leaves its egg.
Determining hatching time is somewhat straightforward, because when the embryo hatches it moves out of the field of view.
In terms of the signal recorded, this manifests as an abrupt drop in both active and structural channel signals.

To identify this signal drop we use the structural channel signal, because it's more stable than the active channel.
The structural channel is first smoothed using `scipy.signal.savgol_filter` and zscored.
Then we calculate the baseline of this signal, as the average of the most frequent bin in the signal's histogram.
The signal used to calculate hatching is the structural channel signal minus the baseline.
As a default threshold we use `Z = 0.35`.
The hatching point is then marked as the first point that reaches the Z score.

Notice that all data after the hatching point should be ignored.
Often times another larva that ecloded earlier will enter an empty field of view, creating a sudden peak in signal activity that does not represent anything biologically.
It can also happen that an embryo is still inside the egg and a larva that already hatched crawls close to it.
It's very rare that this screnario affects the ROI calculation, because we only consider the largest connected area for calculating the ROI, which is usually the VNC.
If it does, then the only option is to remove that embryo.

When loading an experiment for the first time, it's worth it to visualize the structural channel signal of each embryo.
On a few occasions, mostly due to very abrupt motion, the ROI is understimated and the hatching point is determined earlier.
In these cases, you can drag the line that indicates the hatching to a more accurate positon or remove that embryo.

If the default Z-score of 0.35 does not work in your case, you can adjust it to another value.
Inside the GUI, open the Config file `Menu... View pd_params` and change the value of the Z-score variable.