from collections import deque


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
