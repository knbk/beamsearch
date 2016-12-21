import functools
import math
import numpy as np

from scipy.stats import ttest_ind


def weighted_relative_accuracy(x, y, p_subgroup, subgroup):
    p_y = y[p_subgroup]
    sub_y = y[subgroup]
    p1 = np.count_nonzero(p_y)
    p2 = np.count_nonzero(sub_y)
    n1 = np.count_nonzero(~p_y)
    n2 = np.count_nonzero(~sub_y)

    return p2 / p1 - n2 / n1


def semi_elift(x, y, p_subgroup, subgroup):
    p_y = y[p_subgroup]
    sub_y = y[subgroup]
    p1 = np.count_nonzero(p_y)
    p2 = np.count_nonzero(sub_y)
    n1 = np.count_nonzero(~p_y)
    n2 = np.count_nonzero(~sub_y)

    if p2+n2 == 0 or p1+n1 == 0:
        return 0
    else:
        return (n2 / (p2 + n2)) / (n1 / (p1 + n1))


def specificity(x, y, p_subgroup, subgroup):
    p_y = y[p_subgroup]
    sub_y = y[subgroup]

    n1 = np.count_nonzero(~p_y)
    n2 = np.count_nonzero(~sub_y)

    return 1 - n2 / n1


def sensitivity(X, y, p_subgroup, subgroup):
    p_y = y[p_subgroup]
    sub_y = y[subgroup]
    p1 = np.count_nonzero(p_y)
    p2 = np.count_nonzero(sub_y)

    return p2 / p1


def correlation(x, y, p_subgroup, subgroup):
    p_y = y[p_subgroup]
    sub_y = y[subgroup]
    p1 = np.count_nonzero(p_y)
    p2 = np.count_nonzero(sub_y)
    n1 = np.count_nonzero(~p_y)
    n2 = np.count_nonzero(~sub_y)

    if p2 + n2 == 0 or (p1 - p2 + n1 - n2) == 0:
        return 0

    return (p2 * n1 - p1 * n2) / math.sqrt(p1 * n1 * (p2 + n2) * (p1 - p2 + n1 - n2))


def chi_square(x, y, p_subgroup, subgroup):
    p_y = y[p_subgroup]
    p = np.count_nonzero(p_y)
    n = np.count_nonzero(~p_y)

    return (p + n) * (correlation(x, y, p_subgroup, subgroup) ** 2)


def ttest(x, y, p_subgroup, subgroup):
    control = np.logical_and(subgroup, y[:, 0] == 0)
    treatment = np.logical_and(subgroup, y[:, 0] == 1)

    control_y = y[np.where(control)][:, 1]
    treatment_y = y[np.where(treatment)][:, 1]
    if np.count_nonzero(control_y) == 0 or np.count_nonzero(treatment_y) == 0:
        return 0, 0

    measure = ttest_ind(control_y, treatment_y, equal_var=False)
    if measure[1] < 0.05:
        return measure
    else:
        return 0, measure[1]


def negate(f):
    @functools.wraps(f)
    def metric(*args, **kwargs):
        return -f(*args, **kwargs)

    return metric
