import numpy as np
import math
import feature_norm
import reg_type
import data_processor
import reg_trainer.trainer_linear as trainer_linear
import reg_trainer.trainer_logistic as trainer_logistic
import reg_predictor.predictor_linear as predictor_linear
import reg_predictor.predictor_logistic as predictor_logistic


class Regression:
    def __init__(self, setup):
        self.__trained = False
        self.__raw_data = setup.data
        self.__reg_type = setup.regression_type
        self.__test_sample_ratio = setup.test_sample_ratio
        self.__learning_rate = setup.learning_rate
        self.__regularization_lambda = setup.regularization_lambda
        if self.__reg_type == reg_type.RegressionType.logistic:
            self.__output_case_sensitive = setup.output_case_sensitive
        self.__reg_alg = setup.regression_algorithm
        self.__setup_samples()

        if self.__reg_type == reg_type.RegressionType.linear:
            self.__trainer = trainer_linear.RegressionTrainerLinear(
                coefficient_matrix=self.__x_training,
                outputs=self.__y_training,
                regularization_lambda=self.__regularization_lambda)
            print('Linear reg_trainer setup.')
        elif self.__reg_type == reg_type.RegressionType.logistic:
            self.__trainer = trainer_logistic.RegressionTrainerLogistic(
                coefficient_matrix=self.__x_training,
                outputs=self.__y_training,
                regularization_lambda=self.__regularization_lambda,
                output_case_sensitive=self.__output_case_sensitive)
            print('Logistic reg_trainer setup.')

    def train(self,
              time_limit=None,
              iteration_limit=None,
              print_cost=False):
        self.__trained = False
        if not self.__trainer:
            raise AttributeError('Cannot start training, reg_trainer not found.')
        self.__trainer.start_training(training_algorithm=self.__reg_alg,
                                      learning_rate=self.__learning_rate,
                                      time_limit=time_limit,
                                      iteration_limit=iteration_limit,
                                      print_cost_while_training=print_cost)
        self.__trained = True
        self.__setup_predictor()
        self.__print_error_rate()

    def predict(self, data):
        if not self.__predictor:
            raise Exception('Cannot predict, no reg_predictor found.')
        return (self.__predictor.predict(data)).flatten()

    def __setup_samples(self):
        if self.__raw_data is None:
            raise ValueError('Cannot setup samples, no data.')
        num_training_sample = self.__get_training_sample_count()
        self.__x_training = self.__raw_data[:num_training_sample, :-1]
        self.__y_training = self.__raw_data[:num_training_sample, -1]
        self.__x_testing = self.__raw_data[num_training_sample:, :-1]
        self.__y_testing = self.__raw_data[num_training_sample:, -1]
        self.__preprocess_training_set_features()

    def __setup_predictor(self):
        if not self.__trained:
            raise Exception('Cannot setup reg_predictor, model has not been trained.')
        if self.__reg_type == reg_type.RegressionType.linear:
            self.__predictor = predictor_linear.RegressionPredictorLinear(
                weights=self.__trainer.weights,
                feature_normalizer=self.__feature_normalizer)
        elif self.__reg_type == reg_type.RegressionType.logistic:
            self.__predictor = predictor_logistic.RegressionPredictorLogistic(
                weights=self.__trainer.weights,
                feature_normalizer=self.__feature_normalizer,
                categories=self.__trainer.categories)

    def __print_error_rate(self):
        error_rate = self.__get_testing_set_error_rate()
        print('Error rate is {0:.2f}%.'.format(error_rate * 100))

    def __get_training_sample_count(self):
        total_sample_count = np.size(self.__raw_data, axis=0)
        return math.ceil((1.0 - self.__test_sample_ratio) * total_sample_count)

    def __preprocess_training_set_features(self):
        if self.__x_training is None:
            raise ValueError('Cannot pre-process training set features, no data.')
        self.__feature_normalizer = feature_norm.FeatureNormalizer(self.__x_training)
        self.__x_training = self.__feature_normalizer.normalized_feature()
        self.__x_training = data_processor.DataProcessor.add_x0_column(self.__x_training)

    def __get_testing_set_error_rate(self):
        if not self.__predictor:
            raise Exception('Cannot get error rate, no reg_predictor found.')
        testing_sample_predictions = self.predict(self.__x_testing)
        return self.__get_error_rate(testing_sample_predictions, self.__y_testing)

    def __get_error_rate(self, prediction, actual):
        if self.__reg_type == reg_type.RegressionType.linear:
            diff = np.abs((prediction - actual) / actual)
            diff = diff[~np.isnan(diff)]
            return np.average(diff)
        elif self.__reg_type == reg_type.RegressionType.logistic:
            match = prediction == actual
            assert isinstance(match, np.ndarray)
            match_count = np.count_nonzero(match)
            total_count = np.size(actual)
            return (total_count - match_count) / total_count
