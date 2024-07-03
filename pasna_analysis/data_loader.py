from pathlib import Path


class DataLoader:
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

    def activities(self) -> list[Path]:
        '''Returns a list of activity csv files.'''
        activity_dir = self.path.joinpath('activity')
        return sorted(list(activity_dir.iterdir()), key=emb_id)

    def lengths(self) -> list[Path]:
        '''Returns a list of VNC length csv files.'''
        length_dir = self.path.joinpath('lengths')
        return sorted(list(length_dir.iterdir()), key=emb_id)

    def get_embryo_files_by_id(self, id: int) -> tuple[Path]:
        '''Returns a tuple with activity and length files for a given `id`.

        If no files are found, returns None for each file.'''
        emb = f"emb{id}"
        a = next((e for e in self.activities() if e.stem == emb), None)
        l = next((e for e in self.lengths() if e.stem == emb), None)
        return a, l


def emb_id(emb: Path | str) -> int:
    '''Assumes that embryos are always named as emb + id, e.g: `emb21`.'''
    if isinstance(emb, Path):
        emb = emb.stem
    return int(emb[3:])
