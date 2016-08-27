import numpy as np
import regression.regpkg_shuyang.reg_alg as reg_alg
import regression.regpkg_shuyang.reg_trainer.trainer_super as trainer_super


# noinspection PyClassHasNoInit
class RegressionTrainerLinear(trainer_super.RegressionTrainer):
    @staticmethod
    def __get_feature_count_threshold() -> int:
        return 10000

    # Override (abstract)
    def _setup_training(self):
        # noinspection PyProtectedMember
        super()._setup_training()
        self._theta = np.zeros(np.size(self._x, axis=1))

    # Override (abstract)
    def _calculate_optimized_training_alg(self):
        feature_count_small = self._get_num_features() < self.__get_feature_count_threshold()
        if feature_count_small:
            return reg_alg.RegressionAlgorithm.normal_equation
        else:
            return reg_alg.RegressionAlgorithm.gradient_descent

    # Override (abstract)
    def _hypothesis(self):
        return self._theta @ self._x.transpose()

    # Override (abstract)
    def _train(self,
               training_algorithm=reg_alg.RegressionAlgorithm.unspecified,
               learning_rate=0.01,
               time_limit=None,
               iteration_limit=None,
               print_cost_while_training=False):
        if training_algorithm == reg_alg.RegressionAlgorithm.gradient_descent:
            self._train_with_gradient_descent(learning_rate, time_limit, iteration_limit, print_cost_while_training)
        elif training_algorithm == reg_alg.RegressionAlgorithm.normal_equation:
            self.__train_with_normal_equation()
        else:
            raise ValueError('Cannot start training, no linear regression algorithm specified.')

    def __train_with_normal_equation(self):
        x = self._x
        x_trans = x.transpose()
        y = self._y
        regularization_matrix = np.identity(self._get_num_features())
        regularization_matrix[0][0] = 0
        regularization_matrix *= self._regularization_lambda
        try:
            result = np.linalg.inv(x_trans @ x + regularization_matrix) @ x_trans @ y
        except ValueError as e:
            raise Exception('Cannot calculate weights with normal equation.') from e
        else:
            self._theta = result
