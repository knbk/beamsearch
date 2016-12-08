import functools
import numpy as np


def weighted_relative_accuracy(X, y, subgroup):
    subY = y[subgroup]
    P = np.count_nonzero(y)
    p = np.count_nonzero(subY)
    N = np.count_nonzero(~y)
    n = np.count_nonzero(~subY)

    return p/P - n/N


def negate(f):
    @functools.wraps(f)
    def metric(X, y, subgroup):
        return -f(X, y, subgroup)

    return metric
