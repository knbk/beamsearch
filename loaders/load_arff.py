import arff
import numpy as np
from sklearn import preprocessing

data_path = './dataset.arff'

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


class DataModel:
    def __init__(self, x, y, categorical=None, attributes=None):
        self.x = x
        self.y = y
        self.categorical = categorical
        self.attributes = attributes

    def transform(self):
        self.x = self.x[:, keep]
        self.categorical = np.array(self.categorical)
        self.categorical = self.categorical[keep]
        numerical = ~self.categorical
        Xnum = self.x[:, numerical]
        Xnum = preprocessing.Imputer(strategy='median').fit_transform(Xnum, self.y)
        Xcat = self.x[:, self.categorical]
        Xcat = preprocessing.Imputer(strategy='most_frequent').fit_transform(Xcat, self.y)
        self.x[:, numerical] = Xnum
        self.x[:, self.categorical] = Xcat
        self.attributes = [x for i, x in enumerate(self.attributes) if keep[i]]


def load_data(path, include_categorical=False, include_attributes=False):
    path = path or data_path

    with open(path, 'r') as fp:
        data = arff.load(fp, encode_nominal=True)

    np_data = np.array(data['data'])
    x = np_data[:, 0:-3]
    y = np_data[:, -3:]

    categorical = [isinstance(type_, list) for _, type_ in data['attributes'][:-3]]
    attributes = data['attributes'][:-3]

    ret_val = DataModel(x, y)
    if include_categorical:
        ret_val.categorical = categorical
    if include_attributes:
        ret_val.attributes = attributes

    return ret_val


def get_data(path=None):
    data = load_data(path, True, True)
    data.transform()
    return data
