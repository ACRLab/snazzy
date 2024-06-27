from collections import deque
from pathlib import Path


class CounterDeque(deque):
    '''A deque that memoizes how many values were popped and the coordinates
    of the extreme values.'''

    def __init__(self, shape):
        super().__init__()
        self.counter = 0
        self.extremes = [shape[0], 0, shape[1], 0]

    def pop(self):
        self.counter += 1
        return super().pop()

    def update_extremes(self, i, j):
        '''Compares list values with `i` and `j`, keeps the highest values.'''
        min_i, max_i, min_j, max_j = self.extremes
        if i < min_i:
            self.extremes[0] = i
        if i > max_i:
            self.extremes[1] = i
        if j < min_j:
            self.extremes[2] = j
        if j > max_j:
            self.extremes[3] = j


def emb_number(emb_path: Path | str) -> str:
    '''Assumes that embryos are always named as embXX-chY.

    Sorts by the embryo number (XX in the examble above).'''
    if isinstance(emb_path, Path):
        emb_path = emb_path.stem
    return int(emb_path.split('-')[0][3:])


def emb_name(number: int, ch: int) -> str:
    '''Returns the embryo name for a given embryo number.'''
    return f'emb{number}-ch{ch}'


def format_seconds(seconds):
    '''Returns HH:mm:ss, given an amount of seconds.'''
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_time = '{:02}:{:02}:{:02}'.format(
        int(hours), int(minutes), int(seconds))
    return formatted_time
