from pasna_analysis import Experiment


class Group:
    '''A group of experiments that should be analyzed together.'''

    def __init__(self, name: str, experiments: dict[str, Experiment]):
        self.name = name
        self.experiments = experiments
