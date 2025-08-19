import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from gowf_analysis import Config, Trace


class Embryo:
    """Encapsulates data about a given embryo."""

    def __init__(
        self,
        activity_data: np.ndarray,
        vnc_len_data: np.ndarray,
        emb_size_data: np.ndarray,
        name: str,
        config: Config,
    ):
        self.name = name
        self.activity = activity_data
        self.vnc_length = vnc_len_data
        self.vnc_length_filtered = gaussian_filter1d(self.vnc_length[:, 1], sigma=3)
        self.size = self.get_emb_size(emb_size_data)
        self.dev_time_interpolator = None
        self.time_interpolator = None
        self.trace = Trace(
            self.name,
            self.activity,
            config=config,
        )
        self._lin_dev_time = None

    @property
    def lin_developmental_time(self):
        if self._lin_dev_time is None:
            return self._lin_developmental_time()
        return self._lin_dev_time

    def developmental_time(self) -> np.ndarray:
        """Returns emb_size:VNC_size ratio."""
        return self.size / self.vnc_length_filtered

    def _lin_developmental_time(self) -> np.ndarray:
        """Returns the linearized (1st deg polynomial fit) developmental time."""
        vnc_time = self.vnc_length[:, 0]
        linear_fit = Polynomial.fit(x=vnc_time, y=self.developmental_time(), deg=1)
        activity_time = self.activity[:, 0]
        self._lin_dev_time = linear_fit(activity_time)
        return self._lin_dev_time

    def get_time_bins(self, bins):
        """Given an array of bins in developmental time, return the corresponding time bins."""
        lin_dev_time = self.lin_developmental_time
        insert_idxs = np.searchsorted(lin_dev_time, bins)
        bin_idxs = np.unique(insert_idxs)
        bin_idxs[-1] -= 1
        time_bins = self.trace.time[bin_idxs]
        idx_offset = np.count_nonzero(insert_idxs == 0) - 1
        return time_bins, idx_offset

    def get_DT_from_time(self, time: np.ndarray | float) -> np.ndarray | float:
        """Returns the estimated (by linear interpolation) developmental time
        for a time series."""
        if self.dev_time_interpolator is None:
            dvt = self.developmental_time()
            dvt_timepoints = self.vnc_length[:, 0]
            dev_time_interp = interp1d(
                dvt_timepoints, dvt, kind="linear", fill_value="extrapolate"
            )
            self.dev_time_interpolator = dev_time_interp
        dt = self.dev_time_interpolator(time)
        if dt.size == 1:
            return dt.item()
        return dt

    def get_time_from_DT(self, dev_time: np.ndarray | float) -> np.ndarray | float:
        """Returns the estimated (by linear interpolation) time given a developmental time sequence."""
        if self.time_interpolator is None:
            dvt = self.developmental_time()
            dvt_timepoints = self.vnc_length[:, 0]
            time_interp = interp1d(
                dvt, dvt_timepoints, kind="linear", fill_value="extrapolate"
            )
            self.time_interpolator = time_interp
        time = self.time_interpolator(dev_time)
        if time.size == 1:
            return time.item()
        return time

    def get_id(self) -> int:
        """Returns the number that identifies an embryo."""
        return int(self.name[3:])

    def get_emb_size(self, emb_size_data: np.ndarray) -> float:
        """Extracts embryo size."""
        id = self.get_id()
        emb = emb_size_data[emb_size_data[:, 0] == id]
        return emb[0, 1]
