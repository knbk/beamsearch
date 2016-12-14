from beamsearch import DataModel
import numpy as np


def find_discrimination(data):
    """
    :type data: DataModel
    """
    simplify_data(data)
    p, n, original_fraction = count_set(data)
    results = [("complete_set", p, n, original_fraction, 1.0)]

    for i in range(data.x.shape[1]):
        attr = data.attributes[i]
        for j in range(len(attr[1])):
            value = attr[1][j]
            p, n, fraction = count_set(data, (i, j))
            identifier = str(attr[0]) + "_" + value
            if n + p > 500:
                results.append((identifier, p, n, fraction, fraction / original_fraction))

    np_results = np.array(results)
    with open('discrimination_significant_sizes.csv', 'w') as file:
        for i in range(len(np_results)):
            row = np_results[i]
            for j in range(len(row)):
                file.write(str(row[j]))
                if j + 1 < len(row):
                    file.write(";")
            if i + 1 < len(np_results):
                file.write("\n")


def count_attr_values(attributes):
    count = 0
    for attr in attributes:
        count += len(attr[1])
    return count


def count_set(data, bound=None):
    """
    :type bound: {int, string}
    :type data: DataModel
    """
    p = 0
    n = 0
    if bound == None:
        for i in data.y:
            if i[2] == 0:
                n += 1
            else:
                p += 1
    else:
        for i in range(data.x.shape[0]):
            if data.x[i, bound[0]] == bound[1]:
                if data.y[i][2] == 0:
                    n += 1
                else:
                    p += 1
    if n == 0 & p == 0:
        return p, n, None
    else:
        return p, n, p / n


def simplify_data(data):
    """
    :type data: DataModel
    """
    for i in range(data.x.shape[1]):
        if data.attributes[i][1] == 'NUMERIC':
            new_attribute = data.attributes[i][0], split_to_buckets(data.x, i, 5)
            data.attributes[i] = new_attribute


def split_to_buckets(x, c, n):
    """
    :param x: the table (x)
    :param c: the column of the table
    :param n: the count of buckets
    """
    maximum = - np.infty
    minimum = np.infty

    for r in range(x.shape[0]):
        if x[r, c] > maximum:
            maximum = x[r, c]
        if x[r, c] < minimum:
            minimum = x[r, c]

    maximum += 1
    step = np.ceil((maximum - minimum) / n)
    buckets = []
    for bucket in range(n):
        buckets.append("[" + str(minimum + (bucket * step)) + ":" + str(minimum + ((bucket + 1) * step) - 1) + "]")

    if step != 0:
        for r in range(x.shape[0]):
            old_value = x[r, c]
            bucket = int((old_value - minimum) / step)
            if bucket >= len(buckets):
                print('error')
            x[r, c] = bucket - 1
        return buckets
    return [min]
