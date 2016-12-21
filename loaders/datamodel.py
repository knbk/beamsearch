import re

import numpy as np
import sys


class DataModel:
    """
    :type x: np.ndarray
    :type y: np.ndarray
    :type attributes: list
    :type categorical: list
    :type target_index: int | slice
    """

    def __init__(self, x, y, categorical=None, attributes=None, attribute_types=None):
        self.x = x
        self.y = y
        self.categorical = categorical
        self.attributes = attributes
        self.attribute_types = attribute_types
        self.target_index = None
        self.target_attributes = []

    def set_target_index(self, index):
        """
        :type index: int | slice | None
        :param index: The index to get y from.
        :return: None
        """
        if index != self.target_index:
            if isinstance(index, slice):
                if index.step is not None:
                    raise ValueError("Cannot set target index to a slice with step parameter.")
            elif index is not None:
                index = slice(index, index + 1)

            if self.target_index is not None:
                self.x = np.concatenate((self.x[:, :self.target_index.start], self.y,
                                         self.x[:, self.target_index.start:]), axis=1)
                self.attributes = (self.attributes[:self.target_index.start] + self.target_attributes
                                   + self.attributes[self.target_index.start:])
                self.y = None
                self.target_attributes = None
                self.target_index = None

            if index is not None:
                self.target_index = index
                self.y = self.x[:, index]
                self.target_attributes = self.attributes[index]
                self.x = np.concatenate((self.x[:, :index.start], self.x[:, index.stop:]), axis=1)
                self.attributes = self.attributes[:index.start] + self.attributes[index.stop:]

    def encode_values(self):
        """
        Encodes values
        Should only be done when its never done before

        :return: None
        """
        if self.target_attributes is not None:
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
