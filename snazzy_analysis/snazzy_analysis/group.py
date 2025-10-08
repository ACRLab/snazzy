from dataclasses import dataclass

from snazzy_analysis import Dataset


@dataclass
class Group:
    """A group of datasets that should be analyzed together."""

    name: str
    datasets: dict[str, Dataset]

    @property
    def number_of_embryos(self):
        """Total number of embryos across all datasets in this group."""
        total_embs = 0
        for dataset in self.datasets.values():
            total_embs += len(dataset.embryos)
        return total_embs
