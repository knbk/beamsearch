import functools
import math
import numpy as np


def weighted_relative_accuracy(X, y, subgroup):
    subY = y[subgroup]
    P = np.count_nonzero(y)
    p = np.count_nonzero(subY)
    N = np.count_nonzero(~y)
    n = np.count_nonzero(~subY)

    return p/P - n/N


def specificity(X, y, subgroup):
    subY = y[subgroup]

    N = np.count_nonzero(~y)
    n = np.count_nonzero(~subY)

    return 1 - n/N


def sensitivity(X, y, subgroup):
    subY = y[subgroup]
    P = np.count_nonzero(y)
    p = np.count_nonzero(subY)

    return p/P


def correlation(X, y, subgroup):
    subY = y[subgroup]
    P = np.count_nonzero(y)
    p = np.count_nonzero(subY)
    N = np.count_nonzero(~y)
    n = np.count_nonzero(~subY)

    if p + n == 0 or (P - p + N - n) == 0:
        return 0

    return (p * N - P * n) / math.sqrt(P * N * (p + n) * (P - p + N - n))


def chi_square(X, y, subgroup):
    P = np.count_nonzero(y)
    N = np.count_nonzero(~y)

    return (P + N) * (correlation(X, y, subgroup) ** 2)


def negate(f):
    @functools.wraps(f)
    def metric(X, y, subgroup):
        return -f(X, y, subgroup)

    return metric
