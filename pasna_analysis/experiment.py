from pasna_analysis import DataLoader, Embryo, Trace


class Experiment:
    '''Encapsulates data about all embryos for a given experiment.'''

    def __init__(self, data: DataLoader, first_peak_threshold=30):
        activities = data.activities()
        lengths = data.lengths()
        self.name = data.name
        self.embryos = [Embryo(a, l) for a, l in zip(activities, lengths)]
        self.first_peak_threshold = first_peak_threshold
        self.traces = {}
        self.filter_embryos()

    def filter_embryos(self):
        '''Keeps only the embryos with valid traces.

        A trace is valid if the first peak happens after `min` minutes.'''
        for emb in self.embryos:
            trace = self.get_trace(emb)
            if trace:
                self.traces[emb.name] = trace

        self.embryos = [
            e for e in self.embryos if e.name in self.traces.keys()]

    def get_trace(self, emb: Embryo):
        '''Returns the activity trace for an embryo.'''
        time = emb.activity[:, 0]
        act = emb.activity[:, 1]
        stc = emb.activity[:, 2]

        trace = Trace(time, act, stc)
        first_peak = trace.get_first_peak_time() / 60
        if first_peak < self.first_peak_threshold:
            print(
                f'First peak detected before {self.first_peak_threshold} mins for {emb.name} (t={first_peak} mins). Skipping..')
            return None
        return trace
