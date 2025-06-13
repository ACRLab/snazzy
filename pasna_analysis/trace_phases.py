from matplotlib import patches
from scipy.spatial.distance import pdist, squareform
from skimage.filters import threshold_multiotsu
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class TracePhases:
    """Calculate phase boundaries for a Trace.

    A phase is a time inverval where peaks have similar features. This class is used to
    calculate features and determine when a new phase starts base on feature distances.
    """

    def __init__(self, trace):
        self.trace = trace

    def get_phase1_end(self) -> int:
        """Return the index of the last phase 1 peak."""
        features = self.phase1_features()
        dm = TracePhases.dist_matrix(features)
        thres = TracePhases.feature_thres(dm)
        p1_end = TracePhases.apply_threshold_to_matrix(dm, thres)

        return p1_end

    def get_dsna_start(self) -> int:
        """Return the index where dSNA starts.

        Used for specific traces where this behavior is observed, eg vgludf."""
        features = self.dsna_features()
        if len(features) == 0:
            return -1
        dm = TracePhases.dist_matrix(features)
        thres = TracePhases.feature_thres(dm)
        dsna_start = TracePhases.apply_threshold_to_matrix(dm, thres, reverse=True)

        return self.trace.peak_idxes[dsna_start]

    def phase1_features(self, hf_cutoff: float = 0.025) -> list:
        """Each peak is represented by HF pass RMS and peak amplitude.

        Parameters:
            hf_cutoff(float):
                Higher frequency cutoff. Calculates peak RMS after removing all
                frequencies lower than this value.
        """
        high_pass = self.trace.get_filtered_signal(hf_cutoff, low_pass=False)

        features = []
        for pi, (s, e) in zip(self.trace.peak_idxes, self.trace.peak_bounds_indices):
            feature = []
            rms = np.sqrt(np.mean(np.power(high_pass[s:e], 2)))
            feature.append(self.trace.dff[pi])
            feature.append(rms)
            features.append(feature)

        return features

    def dsna_features(self, lf_cutoff: float = 0.002) -> list:
        """Each peak is represented by LF pass RMS and peak amplitude.

        Parameters:
            lf_cutoff(float):
                Lower frequency cutoff. Calculates peak RMS after removing all
                frequencies higher than this value.
        """
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

    @staticmethod
    def feature_thres(dist_matrix, num_classes: int = 2) -> float:
        """Return the distance threshold using threshold multiotsu.

        Parameters:
            dist_matrix(ndarray):
                Square matrix of distance features.
            num_classes(int):
                Number of classes to use in threshold_multiotsu. The threshold
                returned is the highest threshold found by threshold_multiotsu.
                Using more classes tends to increate the threshold value.
        """
        thres_vals = threshold_multiotsu(dist_matrix, classes=num_classes)
        return thres_vals[-1]

    @staticmethod
    def dist_matrix(features: list) -> np.ndarray:
        """Return a square matrix of feature distances.

        Features are first minMax scaled before calculating distances.

        Parameters:
            features(list):
                A list where each element represents a list of features for a
                condition.
        """
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        return squareform(pdist(features_scaled, metric="euclidean"))

    @staticmethod
    def apply_threshold_to_matrix(matrix, thres, reverse=False) -> int:
        """Return index of the first cell in the matrix above `thres`.

        For each cell in the matrix diagonal, calculate the average distance of
        all previous cells. Stop when the average reaches the threshold.

        Based on the iteration order, the returned value has different meanings:

        When reverse == `False`, the index returned is the last cell of that phase.
        When reverse == `True`, the index returned is the first cell of that phase.

        Parameters:
            matrix (nparray):
                Feature distances square matrix.
            thres (float):
                Threshold used to determine index.
            reverse (bool):
                Determines the iteration order. If `False`, starts from first cell
                and averages distances of previous cells. If `True`, start from
                last cell and averages next cells.
        """
        if matrix.size == 0:
            raise ValueError("Cannot apply threshold to empty matrix.")
        N = len(matrix)
        if N == 1:
            return 0

        if reverse:
            for k in range(N - 2, -1, -1):
                next_cells = matrix[k, k + 1 : N]
                if np.average(next_cells) > thres:
                    return k
        else:
            for k in range(1, N + 1):
                next_cells = matrix[k, :k]
                if np.average(next_cells) > thres:
                    return k - 1
        return N - 1

    def plot_phase_change(self, dist_matrix: np.ndarray, change_index: int):
        """Visualize change index with DFF trace and feat dist matrix."""
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
