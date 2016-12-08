#!/usr/bin/env python
import arff
import os
import numpy as np
from sklearn import preprocessing

data_path = os.path.expanduser('~/.openml/cache/datasets/40536/dataset.arff')


def load_data(include_categorical=False, include_attributes=False):
    with open(data_path, 'r') as fp:
        data = arff.load(fp, encode_nominal=True)

    np_data = np.array(data['data'])
    X = np_data[:, 0:-3]
    y = np_data[:, -1]

    categorical = [isinstance(type_, list) for _, type_ in data['attributes'][:-3]]
    attributes = [attr for attr, _ in data['attributes'][:-3]]

    ret_val = [X, y]
    if include_categorical:
        ret_val.append(categorical)
    if include_attributes:
        ret_val.append(attributes)

    return ret_val


def transform(X, y, categorical, attributes):
    categorical = np.array(categorical)
    numerical = ~categorical
    Xnum = X[:, numerical]
    Xnum = preprocessing.Imputer(strategy='median').fit_transform(Xnum, y)
    Xcat = X[:, categorical]
    Xcat = preprocessing.Imputer(strategy='most_frequent').fit_transform(Xcat, y)
    X[:, numerical] = Xnum
    X[:, categorical] = Xcat
    return X, y, categorical, attributes


def get_data():
    X, y, categorical, attributes = load_data(True, True)
    return transform(X, y, categorical, attributes)