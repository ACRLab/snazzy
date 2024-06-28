from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


class Embryo:
    '''Encapsulates data about a given embryo.'''

    def __init__(self, activity_csv: Path, vnc_len_csv: Path):
        if activity_csv.stem != vnc_len_csv.stem:
            raise ValueError(
                'CSV files for activity and VNC length should refer to the same embryo.')
        self.name = activity_csv.stem
        self.activity = self.import_data(activity_csv)
        self.vnc_length = self.import_data(vnc_len_csv)
        self.vnc_length_filtered = gaussian_filter1d(
            self.vnc_length[:, 1], sigma=3)
        emb_size_csv = vnc_len_csv.parents[1].joinpath('full-length.csv')
        self.size = self.get_emb_size(emb_size_csv)
        self.interpolator = None

    def import_data(self, csv_path: Path) -> np.ndarray:
        return np.loadtxt(csv_path, delimiter=',', skiprows=1)

    def developmental_time(self) -> np.ndarray:
        '''Returns emb_size:VNC_size ratio.'''
        return self.size / self.vnc_length_filtered

    def get_DT_from_time(self, time: float) -> np.ndarray | float:
        '''Returns the estimated (by linear interpolation) developmental time 
        for a time point.'''
        if self.interpolator is None:
            dvt = self.developmental_time()
            dvt_timepoints = self.vnc_length[:, 0]
            interpolator = interp1d(dvt_timepoints, dvt,
                                    kind='linear', fill_value='extrapolate')
            self.interpolator = interpolator
        dt = self.interpolator(time)
        if dt.size == 1:
            return dt.item()
        return dt

    def get_id(self) -> int:
        '''Returns the number that identifies an embryo.'''
        return int(self.name[3:])

    def get_emb_size(self, csv_path: Path) -> np.ndarray:
        '''Extracts embryo size.'''
        id = self.get_id()
        emb_sizes = self.import_data(csv_path)
        emb = emb_sizes[emb_sizes[:, 0] == id]
        return emb[0, 1]
