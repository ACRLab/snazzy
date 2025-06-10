from matplotlib import patches
from scipy.spatial.distance import pdist, squareform
from skimage.filters import threshold_multiotsu, threshold_otsu
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pasna_analysis import Experiment, Trace, utils


class TraceCategories:

    def __init__(self, trace: Trace):
        self.trace = trace

    def get_phase2_start(self):
        """Combines methods to calculate phase 2 start.

        Returns the index of the first peak labeled as phase 2."""
        features = self.phase1_features()
        dm = self.dist_matrix(features)
        thres = self.feature_thres(dm)
        p2_start = self.apply_threshold_to_matrix(dm, thres)

        return p2_start

    def get_dsna_start(self):
        features = self.dsna_features()
        dm = self.dist_matrix(features)
        thres = self.feature_thres(dm)
        dsna_start = self.apply_threshold_to_matrix(dm, thres, reverse=True)

        return dsna_start

    def phase1_features(self, hf_cutoff=0.025):
        high_pass = self.trace.get_filtered_signal(hf_cutoff, low_pass=False)

        features = []
        for pi, (s, e) in zip(self.trace.peak_idxes, self.trace.peak_bounds_indices):
            feature = []
            rms = np.sqrt(np.mean(np.power(high_pass[s:e], 2)))
            feature.append(self.trace.dff[pi])
            feature.append(rms)
            features.append(feature)

        return features

    def dist_matrix(self, features):
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        return squareform(pdist(features_scaled, metric="euclidean"))

    def apply_threshold_to_matrix(self, matrix, thres, reverse=False):
        """Reverse order is used to target traces with dsna."""
        N = len(matrix)
        if N == 1:
            return 0

        if reverse:
            for k in range(N - 1, -1, -1):
                next_cells = np.concatenate(
                    [matrix[k, k + 1 : N], matrix[k + 1 : N, k]]
                )
                if np.average(next_cells) > thres:
                    return k
        else:
            for k in range(1, N + 1):
                next_cells = np.concatenate([matrix[k, :k], matrix[:k, k]])
                if np.average(next_cells) > thres:
                    return k - 1
        return N - 1

    def plot_phase_change(self, dist_matrix, change_index):
        peaks = self.trace.peak_idxes
        peak_amps = self.trace.dff[peaks]
        peak_times = self.trace.time[peaks] / 6

        # plot the mid point between last phase 1 point and first phase 2 point
        try:
            boundary = (peaks[change_index] + peaks[change_index + 1]) // 2
        except IndexError:
            print("Could not determine phase 2 start")
            return

        fig = plt.figure(figsize=(14, 10))
        axs = fig.subplot_mosaic([["dff"], ["dist_matrix"]])

        axs["dff"].axvline(boundary, color="r")
        axs["dff"].plot(self.trace.dff[: self.trace.trim_idx])
        axs["dff"].plot(peak_times, peak_amps, "m.")

        sns.heatmap(
            dist_matrix,
            cmap="viridis",
            square=True,
            xticklabels=False,
            yticklabels=False,
            ax=axs["dist_matrix"],
        )

        rect = patches.Rectangle(
            (0, 0),
            change_index + 1,
            change_index + 1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axs["dist_matrix"].add_patch(rect)
        axs["dist_matrix"].set_title("Feature distances")

        fig.suptitle(self.trace.name)
        plt.show()

    def feature_thres(self, distance_matrix):
        _, thres = threshold_multiotsu(distance_matrix, classes=3)
        thres = threshold_otsu(distance_matrix)
        return thres

    def dsna_features(self, lf_cutoff=0.002):
        low_pass = self.trace.get_filtered_signal(lf_cutoff, low_pass=True)

        features = []
        for i, pi in enumerate(self.trace.peak_idxes):
            feature = []
            s, e = self.trace.peak_bounds_indices[i]
            rms = np.sqrt(np.mean(np.power(low_pass[s:e], 2)))
            feature.append(self.trace.dff[pi])
            feature.append(rms)
            features.append(feature)

        return features
