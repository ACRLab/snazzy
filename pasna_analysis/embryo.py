from pathlib import Path
import numpy as np


class Embryo:
    '''Encapsulates data about a given embryo.'''

    def __init__(self, activity_csv: Path, vnc_len_csv: Path):
        self.name = activity_csv.stem
        self.activity = self.import_data(activity_csv)
        self.vnc_length = self.import_data(vnc_len_csv)
        emb_size_csv = vnc_len_csv.parents[1].joinpath('full-length.csv')
        self.size = self.get_emb_size(emb_size_csv)

    def import_data(self, csv_path) -> np.ndarray:
        return np.loadtxt(csv_path, delimiter=',', skiprows=1)

    def developmental_time(self) -> np.ndarray:
        '''Returns emb_size:VNC_size ratio.'''
        return self.size / self.vnc_length[:, 1]

    def get_id(self) -> int:
        '''Returns the number that identifies an embryo.'''
        return int(self.name[3:])

    def get_emb_size(self, csv_path) -> np.ndarray:
        '''Extracts embryo size.'''
        id = self.get_id()
        emb_sizes = self.import_data(csv_path)
        emb = emb_sizes[emb_sizes[:, 0] == id]
        return emb[0, 1]
