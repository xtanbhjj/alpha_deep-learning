#! coding: utf-8

import time
import numpy as np
import torch


class Timer:
    # Record multiple time points
    def __init__(self):
        self._times = []
        self._tik = time.time()
        self.start()
        pass

    def start(self):
        # Star timer
        self._tik = time.time()

    def stop(self):
        # Stop timer and record time duration.
        self._times.append(time.time() - self._tik)
        return self._times[-1]

    def avg(self):
        # Return average time duration
        return sum(self._times) / len(self._times)

    def sum(self):
        # Return the sum of all time durations
        return sum(self._times)

    def cumsum(self):
        return torch.tensor(self._times).cumsum(dim=-1).tolist()
        # return np.array(self._times).cumsum().tolist()
