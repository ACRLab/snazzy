from matplotlib import patches
from scipy.spatial.distance import pdist, squareform
from skimage.filters import threshold_multiotsu
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from golf_analysis import FrequencyAnalysis


class TracePhases:
    """Calculate phase boundaries for a Trace.

    A phase is a time inverval where peaks have similar features. This class is used to
    calculate features and determine phase boundaries based on feature distances.
    """

    def __init__(self, trace):
        self.trace = trace

    def get_phase1_end(self, freq: float = 0.025) -> int:
        """Return the index of the last phase 1 peak.

        Parameters:
            freq(float):
                High frequency cutoff value used to calculate the HF feature.

        Returns:
            phase1_end(int):
                Time series index of the last phase 1 peak.
        """
        features = self.phase1_features(hf_cutoff=freq)
        if len(features) <= 1:
            return -1
        dm = TracePhases.dist_matrix(features)
        thres = TracePhases.feature_thres(dm, num_classes=3)
        p1_end = TracePhases.segment_distance_matrix_forward(dm, thres)

        return self.to_dff_index(p1_end)

    def get_dsna_start(self, freq: float = 0.002) -> int:
        """Return the index of the first dSNA peak.

        Used for specific traces where this behavior is observed, eg vgludf.

        Parameters:
            freq(float):
                Low frequency cutoff used to calculate the LF feature.

        Returns:
            dsna_start(int):
                Time series index of the first dSNA peak.
        """
        features = self.dsna_features(lf_cutoff=freq)
        if len(features) <= 1:
            return -1

        dm = TracePhases.dist_matrix(features)
        thres = TracePhases.feature_thres(dm)
        dsna_start = TracePhases.segment_distance_matrix_reverse(dm, thres)

        return self.to_dff_index(dsna_start)

    def to_dff_index(self, peak_idx: int):
        """Convert from peak index to DFF index."""
        peak_idxes = self.trace.get_all_peak_idxes()
        return peak_idxes[peak_idx]

    def phase1_features(self, hf_cutoff: float = 0.025) -> list:
        """Each peak is represented by high frequency pass RMS.

        Parameters:
            hf_cutoff(float):
                High frequency cutoff. Calculates peak RMS after removing all
                frequencies lower than this value.

        Returns:
            features(list):
                A 2D list where each nested list has features of a given peak.
                The only phase 1 feature used is HF filtered RMS.
        """
        high_pass = FrequencyAnalysis.get_filtered_signal(
            self.trace.dff, hf_cutoff, low_pass=False
        )

        rel_height = self.trace.pd_params["peak_width"]
        peak_idxes = self.trace.get_all_peak_idxes()
        peak_bounds = self.trace.compute_peak_bounds(rel_height, peak_idxes)
        features = []
        for pi, (s, e) in zip(peak_idxes, peak_bounds):
            rms = np.sqrt(np.mean(np.power(high_pass[s:e], 2)))
            features.append([self.trace.dff[pi], rms])

        return features

    def dsna_features(self, lf_cutoff: float = 0.005) -> list:
        """Each peak is represented by LF pass RMS and peak amplitude.

        Parameters:
            lf_cutoff(float):
                Lower frequency cutoff. Calculates peak RMS after removing all
                frequencies higher than this value.

        Returns:
            features(list):
                A 2D list where each nested list has features of a given peak.
        """
        low_pass = FrequencyAnalysis.get_filtered_signal(
            self.trace.dff, lf_cutoff, low_pass=True
        )

        rel_height = self.trace.pd_params["peak_width"]
        peak_idxes = self.trace.get_all_peak_idxes()
        peak_bounds = self.trace.compute_peak_bounds(rel_height, peak_idxes)
        features = []
        for i, (s, e) in enumerate(peak_bounds):
            rms = np.sqrt(np.mean(np.power(low_pass[s:e], 2)))
            peak_amp = self.trace.dff[peak_idxes[i]]
            features.append([rms, peak_amp])

        return features

    @staticmethod
    def feature_thres(dist_matrix: np.ndarray, num_classes: int = 2) -> float:
        """Return the lower threshold value from multiclass otsu threshold.

        Increasing the number of classes decreases the threshold value.

        Parameters:
            dist_matrix(ndarray):
                Square matrix of distance features.
            num_classes(int):
                Number of classes to use in threshold_multiotsu.
        """
        thres_vals = threshold_multiotsu(dist_matrix, classes=num_classes)
        return thres_vals[0]

    @staticmethod
    def dist_matrix(features: list) -> np.ndarray:
        """Return a square matrix of feature distances.

        To ensure that all features have the same range and contribute equally
        to the distance matrix, they are first minMax scaled before calculating
        distances.

        Parameters:
            features(list):
                A list where each element represents a list of features for a
                condition.
        """
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        return squareform(pdist(features_scaled, metric="euclidean"))

    @staticmethod
    def segment_distance_matrix_forward(matrix: np.ndarray, thres: float) -> int:
        """Segmentation by region growing until the provided thres is reached.

        Iterate over each cell in the matrix diagonal and calculate the average
        distance between that cell and all previous cells. Stop when the average
        reaches the threshold.

        Parameters:
            matrix (nparray):
                Feature distances square matrix.
            thres (float):
                Threshold used to determine index.

        Returns:
            segmentation_index (int):
                Highest index that is still below `thres`.
        """
        if matrix.size == 0:
            raise ValueError("Cannot apply threshold to empty matrix.")

        N = len(matrix)
        if N == 1:
            return 0

        for k in range(1, N):
            next_cells = matrix[k, :k]
            if np.average(next_cells) > thres:
                return k - 1

        return N - 1

    @staticmethod
    def segment_distance_matrix_reverse(matrix: np.ndarray, thres: float) -> int:
        """Segmentation by region growing until the provided thres is reached.

        **Starting from the last element of the matrix**, iterate over each cell
        in the matrix diagonal and calculate the average distance between that
        cell and all forward cells. Stop when the average reaches the threshold.

        Parameters:
            matrix (nparray):
                Feature distances square matrix.
            thres (float):
                Threshold used to determine index.

        Returns:
            segmentation_index (int):
                Lowest index that is still below `thres`.
        """
        if matrix.size == 0:
            raise ValueError("Cannot apply threshold to empty matrix.")

        N = len(matrix)
        if N == 1:
            return 0

        for k in range(N - 2, -1, -1):
            next_cells = matrix[k, k + 1 :]
            if np.average(next_cells) > thres:
                return k + 1

        return N - 1

    def plot_phase_change(
        self,
        dist_matrix: np.ndarray,
        change_index: int,
        features: list,
        from_start: bool = True,
    ):
        """Visualize change index with DFF trace and feat dist matrix."""
        peaks = self.trace.peak_idxes
        peak_amps = self.trace.dff[peaks]
        peak_times = self.trace.time[peaks] / 6
        all_peaks = self.trace.get_all_peak_idxes()

        # plot the mid point between last phase 1 point and first phase 2 point
        try:
            boundary = (all_peaks[change_index] + all_peaks[change_index + 1]) // 2
        except IndexError:
            print("Could not determine phase 2 start")
            return

        fig = plt.figure(figsize=(14, 10))
        axs = fig.subplot_mosaic([["dff", "dff"], ["features", "dist_matrix"]])

        axs["dff"].axvline(boundary, color="r")
        axs["dff"].plot(self.trace.dff[: self.trace.trim_idx])
        axs["dff"].plot(peak_times, peak_amps, "m.")

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        feat1, feat2 = list(zip(*scaled_features))
        axs["features"].plot(
            feat1, feat2, linestyle="None", marker="o", markerfacecolor="None"
        )
        axs["features"].set_ylabel("Peak Amplitude")
        axs["features"].set_xlabel("RMS High Frequency Content")

        sns.heatmap(
            dist_matrix,
            cmap="viridis",
            square=True,
            xticklabels=False,
            yticklabels=False,
            ax=axs["dist_matrix"],
        )

        # plot segmented area from start or from finish
        if from_start:
            rect = patches.Rectangle(
                (0, 0),
                change_index + 1,
                change_index + 1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        else:
            width = len(dist_matrix) - change_index
            rect = patches.Rectangle(
                (change_index, change_index),
                width,
                width,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        axs["dist_matrix"].add_patch(rect)
        axs["dist_matrix"].set_title("Feature distances")

        fig.suptitle(self.trace.name)
        plt.show()
