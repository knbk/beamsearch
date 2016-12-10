import arff
import os
import numpy as np
from sklearn import preprocessing

data_path = os.path.expanduser('~/.openml/cache/datasets/40536/dataset.arff')

# This removes all discretized attributes that are also available as numerical attributes.
# Remove these, as the numerical attributes allow for more fine-grained selection.
keep = np.array([
    True, True, True, True, True, True, False, True, True, True, True, True, False, False, True, True, True, True, True,
    True, True, False, False, False, False, False, False, True, True, True, True, True, True, False, False, False,
    False, False, False, True, True, True, True, True, True, False, False, False, False, False, False, True, True, True,
    True, True, False, False, False, False, False, True, True, True, True, True, True, False, False, False, False,
    False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
    False, True, False, True, True, True, False, False, False, True, True, False, False, True
])


class ARFFData:
    def __init__(self, x, y, categorical=None, attributes=None):
        self.x = x
        self.y = y
        self.categorical = categorical
        self.attributes = attributes


def load_data(path, include_categorical=False, include_attributes=False):
    path = path or data_path

    with open(path, 'r') as fp:
        data = arff.load(fp, encode_nominal=True)

    np_data = np.array(data['data'])
    x = np_data[:, 0:-3]
    y = np_data[:, -1]

    categorical = [isinstance(type_, list) for _, type_ in data['attributes'][:-3]]
    attributes = data['attributes'][:-3]

    ret_val = ARFFData(x, y)

    if include_categorical:
        ret_val.categorical = categorical
    if include_attributes:
        ret_val.attributes = attributes

    return ret_val


def transform(X, y, categorical, attributes):
    X = X[:, keep]
    categorical = np.array(categorical)
    categorical = categorical[keep]
    numerical = ~categorical
    Xnum = X[:, numerical]
    Xnum = preprocessing.Imputer(strategy='median').fit_transform(Xnum, y)
    Xcat = X[:, categorical]
    Xcat = preprocessing.Imputer(strategy='most_frequent').fit_transform(Xcat, y)
    X[:, numerical] = Xnum
    X[:, categorical] = Xcat
    attributes = [x for i, x in enumerate(attributes) if keep[i]]
    return X, y, categorical, attributes


def get_data(path=None):
    X, y, categorical, attributes = load_data(path, True, True)
    return transform(X, y, categorical, attributes)
