from pathlib import Path
from typing import Optional

from pasna_analysis import DataLoader, Embryo, Trace


class Experiment:
    '''Encapsulates data about all embryos for a given experiment.'''

    def __init__(self, data: DataLoader, first_peak_threshold=30, to_exclude: Optional[list[int]] = None, dff_strategy='baseline', has_transients=False):
        activities = data.activities()
        lengths = data.lengths()
        self.name = data.name
        self.embryos = [Embryo(a, l) for a, l in zip(activities, lengths)]
        self.first_peak_threshold = first_peak_threshold
        self.traces = {}
        if to_exclude is None:
            to_exclude = []
        self.dff_strategy = dff_strategy
        self.has_transients = has_transients
        self._filter_embryos(to_exclude)

    def _filter_embryos(self, to_exclude: list[int]):
        '''Keeps only the embryos with valid traces.

        A trace is valid if the first peak happens after `first_peak_threshold` minutes.

        Params:
        -------
        to_exclude : list
            List of embryo ids that should be excluded from the experiment.
        '''
        for emb in self.embryos:
            if emb.get_id() in to_exclude:
                continue
            trace = self.get_trace(emb)
            if trace:
                self.traces[emb.name] = trace

        self.embryos = [
            e for e in self.embryos if e.name in self.traces.keys()]

    def get_trace(self, emb: Embryo):
        '''Returns the activity trace for an embryo.

        If no peak is found, returns `None`.'''
        time = emb.activity[:, 0]
        act = emb.activity[:, 1]
        stc = emb.activity[:, 2]

        trace = Trace(time, act, stc, dff_strategy=self.dff_strategy,
                      has_transients=self.has_transients)
        try:
            first_peak = trace.get_first_peak_time() / 60
        except (ValueError, IndexError):
            # if no peak is found, exclude the embryo from the analysis:
            print(f'No peaks detected for {emb.name}. Skipping..')
            return None
        if first_peak < self.first_peak_threshold:
            print(
                f'First peak detected before {self.first_peak_threshold} mins for {emb.name} (t={first_peak} mins). Skipping..')
            return None
        return trace


def get_id_from_filename(filepath: Path) -> int:
    """Parses embryo id from filename.

    Assumes filenames follow the pattern: 'embXX.csv', where XX is the id."""
    return int(filepath.stem[3:])
