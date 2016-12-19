import numpy as np


class DataModel:
    """
    :type x: np.ndarray
    :type y: np.ndarray
    :type attributes: np.ndarray
    :type categorical: np.ndarray
    :type target_index: int
    """

    def __init__(self, x, y, categorical=None, attributes=None):
        self.x = x
        self.y = y
        self.categorical = categorical
        self.attributes = attributes
        self.target_index = None
        self.target_attribute = None

    def set_target_index(self, index):
        if index != self.target_index:
            if self.target_index is not None:
                y = np.reshape(self.y, (self.y.shape[0], 1))
                self.x = np.concatenate((self.x[:, :self.target_index], y, self.x[:, self.target_index:]),
                                        axis=1)
                self.attributes = np.concatenate(
                    (self.attributes[:self.target_index], [self.target_attribute], self.attributes[self.target_index:]))
            self.target_index = index
            self.y = self.x[:, index]
            self.target_attribute = self.attributes[index]
            self.x = np.concatenate((self.x[:, :index], self.x[:, index + 1:]), axis=1)
            self.attributes = np.concatenate((self.attributes[:index], self.attributes[index + 1:]))

    def encode_values(self):
        # TODO: implement encode function.
        print("TODO IMPLEMENT ENCODE FUNCTION")