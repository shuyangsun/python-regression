import numpy as np


class DataProcessor:
    @staticmethod
    def add_x0_column(matrix):
        return np.insert(matrix, obj=0, values=[1], axis=1)

    @staticmethod
    def augmented_to_coefficient_and_b(matrix):
        return matrix[:, :-1], matrix[:, -1]

    @staticmethod
    def partition(matrix, at_ind):
        return matrix[:at_ind], matrix[at_ind:]

    @staticmethod
    def get_unique_categories(output, case_sensitive=True):
        if not case_sensitive:
            output = [x.lower() if isinstance(x, str) else x for x in output]
        return np.unique(output)

    @staticmethod
    def get_unique_categories_and_binary_outputs(output, case_sensitive=True):
        unique_cat = DataProcessor.get_unique_categories(output, case_sensitive)

        if np.size(unique_cat) <= 2:
            outputs_b = np.zeros(np.size(output))
            mask_0 = (output != unique_cat[0])
            mask_1 = (output == unique_cat[0])
            outputs_b[mask_0] = 0
            outputs_b[mask_1] = 1
        else:
            outputs_b = np.tile(output, (np.size(unique_cat), 1))
            for i, cat in enumerate(unique_cat):
                row = outputs_b[i]
                mask_0 = (output != cat)
                mask_1 = (output == cat)
                row[mask_0] = 0
                row[mask_1] = 1

        return unique_cat, outputs_b
