from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig():
    """Provides configuration to create an Experiment."""
    first_peak_threshold: int
    to_exclude: list[int]
    dff_strategy: Literal['baseline', 'local_minima'] = 'baseline'
    has_transients: bool = False
