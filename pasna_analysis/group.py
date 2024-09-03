from dataclasses import dataclass

from pasna_analysis import Experiment


@dataclass
class Group:
    '''A group of experiments that should be analyzed together.'''
    name: str
    experiments: dict[str, Experiment]
