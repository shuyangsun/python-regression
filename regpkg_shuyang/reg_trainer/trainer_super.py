import numpy as np
import time
import abc
import regression.regpkg_shuyang.reg_alg as reg_alg


class RegressionTrainer(metaclass=abc.ABCMeta):
    def __init__(self, coefficient_matrix, outputs, regularization_lambda=0.0):
        self._x = coefficient_matrix
        self._y = outputs
        self._regularization_lambda = regularization_lambda
        self._theta = np.zeros(self._get_num_features())

    @property
    def weights(self):
        return self._theta

    def start_training(self,
                       training_algorithm=reg_alg.RegressionAlgorithm.unspecified,
                       learning_rate=0.01,
                       time_limit=None,
                       iteration_limit=None,
                       print_cost_while_training=False):
        self.__print_start_training_message_and_log_time()

        self._setup_training()
        self.__reset_thetas()

        if not training_algorithm or training_algorithm == reg_alg.RegressionAlgorithm.unspecified:
            training_algorithm = self._calculate_optimized_training_alg()

        print('Training......')

        self._train(training_algorithm,
                    learning_rate,
                    time_limit,
                    iteration_limit,
                    print_cost_while_training)

        self.__print_end_training_message()

    @abc.abstractmethod
    def _setup_training(self):
        print('Setting up reg_trainer......')

    @abc.abstractmethod
    def _calculate_optimized_training_alg(self):
        pass

    @abc.abstractmethod
    def _hypothesis(self):
        pass

    @abc.abstractmethod
    def _train(self,
               training_algorithm=reg_alg.RegressionAlgorithm.unspecified,
               learning_rate=0.01,
               time_limit=None,
               iteration_limit=None,
               print_cost_while_training=False):
        pass

    def _cost(self):
        h_theta_x = self._hypothesis()
        diff = self._y - h_theta_x
        diff_squared = np.power(diff, 2)
        diff_squared_sum = np.sum(diff_squared.transpose(), axis=0)
        theta_squared = np.power(self._theta, 2)
        theta_squared = theta_squared.transpose()
        theta_squared[0] = 0
        theta_squared = theta_squared.transpose()
        theta_squared_sum = np.sum(theta_squared.transpose(), axis=0)
        total = diff_squared_sum + self._regularization_lambda * theta_squared_sum
        return total / (2 * self._get_num_samples())

    def _derivative_of_cost(self):
        h_theta_x = self._hypothesis().astype(np.float64)
        diff = (h_theta_x - self._y).astype(np.float64)
        diff_scaled_with_x_sum = (self._x.transpose() @ diff.transpose()).transpose()
        regularization_vector = self._theta * self._regularization_lambda / self._get_num_samples()
        regularization_vector = regularization_vector.transpose()
        regularization_vector[0] = np.float64(0)
        regularization_vector = regularization_vector.transpose()
        return diff_scaled_with_x_sum / self._get_num_samples() + regularization_vector

    def _train_with_gradient_descent(self,
                                     learning_rate=0.01,
                                     time_limit=None,
                                     iteration_limit=None,
                                     print_cost_while_training=False):
        last_cost = self._cost()
        cost_not_change_count = 0
        cost_check_frequency = 1000
        i = 1
        start_time = time.time()
        condition = True
        # If the cost hasn't changed in 20 (2 * 10) iterations, it converged.
        while condition:
            self._theta -= self._derivative_of_cost() * learning_rate
            # Check and print cost every 10 iterations
            if i == 1 or i % cost_check_frequency == 0:
                current_cost = self._cost()
                if print_cost_while_training:
                    print('Cost of iteration {0}: {1}'.format(i, current_cost))
                try:
                    cost_equal = all(current_cost == last_cost)
                except TypeError:
                    cost_equal = current_cost == last_cost
                if cost_equal:
                    cost_not_change_count += 1
                last_cost = current_cost
            i += 1
            condition = cost_not_change_count < 2
            if time_limit is not None:
                condition = condition and (time.time() - start_time) < time_limit
            if iteration_limit is not None:
                condition = condition and i <= iteration_limit

    def _get_num_features(self):
        return np.size(self._x, axis=1)

    def _get_num_samples(self):
        return np.size(self._x, axis=0)

    def __print_start_training_message_and_log_time(self):
        print('Initializing......')
        self.__training_start_time = time.time()

    def __print_end_training_message(self):
        end_time = time.time()
        print('Used {0:.10f} seconds to train model with {1} samples and {2} features.'.format
              (end_time - self.__training_start_time, self._get_num_samples(), self._get_num_features() - 1))
        print('Training finished.')

    def __reset_thetas(self):
        self._theta.fill(0)
