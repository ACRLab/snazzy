from dataclasses import dataclass

from snazzy_analysis import Experiment


@dataclass
class Group:
    """A group of experiments that should be analyzed together."""

    name: str
    experiments: dict[str, Experiment]

    @property
    def number_of_embryos(self):
        """Total number of embryos across all experiments in this group."""
        total_embs = 0
        for exp in self.experiments.values():
            total_embs += len(exp.embryos)
        return total_embs
