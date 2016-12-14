import functools
import math
import numpy as np


def weighted_relative_accuracy(x, y, subgroup):
    sub_y = y[subgroup]
    p1 = np.count_nonzero(y)
    p2 = np.count_nonzero(sub_y)
    n1 = np.count_nonzero(~y)
    n2 = np.count_nonzero(~sub_y)

    return p2 / p1 - n2 / n1


def semi_elift(x, y, subgroup):
    sub_y = y[subgroup]
    p1 = np.count_nonzero(y)
    p2 = np.count_nonzero(sub_y)
    n1 = np.count_nonzero(~y)
    n2 = np.count_nonzero(~sub_y)
    if p2+n2 == 0 or p1+n1 == 0:
        return 0
    else:
        return (n2 / (p2 + n2)) / (n1 / (p1 + n1))


def specificity(x, y, subgroup):
    sub_y = y[subgroup]

    n1 = np.count_nonzero(~y)
    n2 = np.count_nonzero(~sub_y)

    return 1 - n2 / n1


def sensitivity(X, y, subgroup):
    sub_y = y[subgroup]
    p1 = np.count_nonzero(y)
    p2 = np.count_nonzero(sub_y)

    return p2 / p1


def correlation(x, y, subgroup):
    sub_y = y[subgroup]
    p1 = np.count_nonzero(y)
    p2 = np.count_nonzero(sub_y)
    n1 = np.count_nonzero(~y)
    n2 = np.count_nonzero(~sub_y)

    if p2 + n2 == 0 or (p1 - p2 + n1 - n2) == 0:
        return 0

    return (p2 * n1 - p1 * n2) / math.sqrt(p1 * n1 * (p2 + n2) * (p1 - p2 + n1 - n2))


def chi_square(x, y, subgroup):
    p = np.count_nonzero(y)
    n = np.count_nonzero(~y)

    return (p + n) * (correlation(x, y, subgroup) ** 2)


def negate(f):
    @functools.wraps(f)
    def metric(x, y, subgroup):
        return -f(x, y, subgroup)

    return metric
