import re

import numpy as np
import sys


class DataModel:
    """
    :type x: np.ndarray
    :type y: np.ndarray
    :type attributes: np.ndarray
    :type categorical: np.ndarray
    :type target_index: int
    """

    def __init__(self, x, y, categorical=None, attributes=None, attribute_types=None):
        self.x = x
        self.y = y
        self.categorical = categorical
        self.attributes = attributes
        self.attribute_types = attribute_types
        self.target_index = None
        self.target_attribute = None

    def set_target_index(self, index):
        """
        :type index: int | None
        :param index: The index to get y from.
        :return: None
        """
        if index != self.target_index:
            if self.target_index is not None:
                y = np.reshape(self.y, (self.y.shape[0], 1))
                self.x = np.concatenate((self.x[:, :self.target_index], y, self.x[:, self.target_index:]),
                                        axis=1)
                self.attributes = (self.attributes[:self.target_index] + [self.target_attribute]
                                   + self.attributes[self.target_index:])
                self.y = None
                self.target_attribute = None

            if index is not None:
                self.target_index = index
                self.y = self.x[:, index]
                self.target_attribute = self.attributes[index]
                self.x = np.concatenate((self.x[:, :index], self.x[:, index + 1:]), axis=1)
                self.attributes = self.attributes[:index] + self.attributes[index + 1:]

    def encode_values(self):
        """
        Encodes values
        Should only be done when its never done before

        :return: None
        """
        if self.target_attribute is not None:
            self.set_target_index(None)
        data = self.x
        for column in range(data.shape[1]):
            if self.attribute_types[column] == 'string':
                values = {}
                for row in range(data.shape[0]):
                    value = data[row][column]
                    values.setdefault(value, len(values))
                    data[row][column] = values[value]
                self.attributes[column] = (self.attributes[column], list(values.keys()))
            elif self.attribute_types[column] == 'timeoffset':
                self.attributes[column] = (self.attribute_types[column], 'integer')
                for row in range(data.shape[0]):
                    value = data[row][column]
                    if re.search("\\d+:\\d+\\.\\d+", value):
                        split1 = value.split(':')
                        split2 = split1[1].split('.')
                        data[row][column] = int(split1[0]) * 600 + int(split2[0]) * 10 + int(split2[1])
            elif self.attribute_types[column] == 'bool':
                self.attributes[column] = (self.attributes[column], 'integer')
                for row in range(data.shape[0]):
                    value = data[row][column]
                    if value == 'FALSE':
                        data[row][column] = 0
                    else:
                        if value == 'TRUE':
                            data[row][column] = 1
                        else:  # should never happen
                            data[row][column] = 2
            elif self.attribute_types[column] == 'float':
                self.attributes[column] = (self.attributes[column], 'float')
            else:  # case integer
                self.attributes[column] = (self.attributes[column], 'integer')
