import numpy as np
import math
import regression.regpkg_shuyang.reg_trainer.trainer_super as trainer_super
import regression.regpkg_shuyang.reg_alg as reg_alg
import regression.regpkg_shuyang.data_processor as data_processor


class RegressionTrainerLogistic(trainer_super.RegressionTrainer):
    def __init__(self,
                 coefficient_matrix,
                 outputs,
                 regularization_lambda=0.0,
                 output_case_sensitive=True):
        super().__init__(coefficient_matrix, outputs, regularization_lambda)
        self.__output_case_sensitive = output_case_sensitive

    @property
    def categories(self):
        return self.__categories

    def __get_num_categories(self):
        return np.size(self.__categories)

    # Override
    def _setup_training(self):
        # noinspection PyProtectedMember
        super()._setup_training()
        unique_cat, b_output = data_processor.DataProcessor.\
            get_unique_categories_and_binary_outputs(self._y,
                                                     self.__output_case_sensitive)
        self.__categories = unique_cat
        self._y = b_output
        cat_count = self.__get_num_categories()
        if cat_count < 2:
            raise ValueError('Cannot do logistic regression, there is only one kind of output.')
        elif cat_count == 2:
            self.__binary_classification = True
            self._theta = np.zeros(self._get_num_features())
        else:
            self.__binary_classification = False
            theta_shape = (cat_count, self._get_num_features())
            self._theta = np.zeros(shape=theta_shape)

    # Override (abstract)
    # noinspection PyMethodMayBeStatic
    def _calculate_optimized_training_alg(self):
        return reg_alg.RegressionAlgorithm.gradient_descent

    # Override (abstract)
    def _hypothesis(self):
        theta_transpose_x = self._theta @ self._x.transpose()
        result = np.zeros(shape=(self.__get_num_categories(), self._get_num_samples()))
        result.fill(math.e)
        result **= (-1 * theta_transpose_x)
        result = 1 / (1 + result)
        return result

    # Override (abstract)
    def _train(self,
               training_algorithm=reg_alg.RegressionAlgorithm.unspecified,
               learning_rate=0.01,
               time_limit=None,
               iteration_limit=None,
               print_cost_while_training=False):
        if training_algorithm == reg_alg.RegressionAlgorithm.gradient_descent:
            self._train_with_gradient_descent(learning_rate, time_limit, iteration_limit, print_cost_while_training)
        else:
            raise ValueError('Cannot start training, no logistic regression algorithm specified.')
