import numpy as np


def arg_closest(arr, v):
    return int(np.argmin([abs(x - v) for x in arr]))
