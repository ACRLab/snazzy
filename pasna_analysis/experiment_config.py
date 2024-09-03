from dataclasses import dataclass


@dataclass
class ExperimentConfig():
    """Provides configuration to create an Experiment."""
    first_peak_threshold: int
    to_exclude: list[int]
