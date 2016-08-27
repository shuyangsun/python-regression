import numpy as np


class FeatureNormalizer:
    def __init__(self, data, data_has_x0_column=False):
        self.__data = data.astype(np.float64)
        self.__data_has_x0_column = data_has_x0_column
        self.__scalars = np.ones(np.size(data, axis=1))
        self.__calculate_scalars()

    def normalized_feature(self):
        return self.normalize_new_feature(self.__data, self.__data_has_x0_column)

    def normalize_new_feature(self, data, input_has_x0_column=False):
        if input_has_x0_column:
            avg = np.insert(self.__avg, obj=0, values=[0])
            std = np.insert(self.__std, obj=0, values=[1])
        else:
            avg = self.__avg
            std = self.__std
        return (data - avg) / std

    def __calculate_scalars(self):
        if self.__data_has_x0_column:
            self.__avg = np.average(self.__data[:, 1:], axis=0)
            self.__std = np.std(self.__data[:, 1:], axis=0)
        else:
            self.__avg = np.average(self.__data, axis=0)
            self.__std = np.std(self.__data, axis=0)
