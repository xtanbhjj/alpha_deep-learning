#! coding: utf-8


class Accumulator:
    def __init__(self, n):
        self._data = [0.0] * n

    def add(self, *args):
        self._data = [a + float(b) for a, b in zip(self._data, args)]

    def reset(self):
        self._data = [0.0] * len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
