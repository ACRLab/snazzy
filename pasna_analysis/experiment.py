from pathlib import Path


def emb_id(emb: Path | str) -> int:
    '''Assumes that embryos are always named as emb + id, e.g: `emb21`.'''
    if isinstance(emb, Path):
        emb = emb.stem
    return int(emb[3:])


class Experiment:
    '''Used to access data about the current experiment.

    Relies on the folder structure described in this project README.'''

    def __init__(self, path: Path):
        if not isinstance(path, Path):
            raise TypeError('Expected a `Path` instance.')
        self.path = path
        self.name = path.stem

    def embryos(self) -> list[str]:
        '''Returns a list of available embryos.'''
        activity_dir = self.path.joinpath('activity')
        return sorted([e.stem for e in activity_dir.iterdir()], key=emb_id)

    def activity(self) -> list[Path]:
        '''Returns a list of activity csv files.'''
        activity_dir = self.path.joinpath('activity')
        return sorted(list(activity_dir.iterdir()), key=emb_id)

    def lengths(self) -> list[Path]:
        '''Returns a list of VNC length csv files.'''
        length_dir = self.path.joinpath('lengths')
        return sorted(list(length_dir.iterdir()), key=emb_id)

    def get_embryo_files_by_id(self, id) -> list[Path]:
        emb = f"emb{id}"
        a = next((e for e in self.activity() if e.stem == emb), None)
        l = next((e for e in self.lengths() if e.stem == emb), None)
        if a and l:
            return (a, l)
        else:
            return None, None
