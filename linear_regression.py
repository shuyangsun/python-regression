import numpy as np
import pandas as pd
import math
import time
import enum

class FeaturePreProcessingType(enum.Enum):
    none = 0
    feature_scaling = 1
    learning_rate_scaling = 2

class Trainer:
    """
    This class is used to perform linear regression.
    """
    def __init__(self,
                 data,
                 test_sample_ratio=0.0,
                 learning_rate=0.01,
                 features_pre_processing_type=FeaturePreProcessingType.feature_scaling):
        """
        Initialize a trainer with a training set data, which is an augmented matrix.
        Test sample percentage indicates how many percent of the data should be used as test samples,
        the rest of them will be training samples.
        """
        if not 0 <= test_sample_ratio < 1:
            raise ValueError('Test sample ratio has to be between greater than or equal to 0, and less than 1.')

        self.__data = data
        self.__learning_rate = learning_rate
        self.__test_sample_ratio = test_sample_ratio
        self.__features_pre_processing_type = features_pre_processing_type

        self.__setup_training_and_testing_sets()
        self.__setup_weights()

    def train(self, print_cost_while_training=False):
        print('Started training...')
        start_time = time.time()
        last_cost = self.__cost_of_training_set()
        cost_not_change_count = 0
        while cost_not_change_count <= 10:
            change = self.__derivative_of_cost()
            if self.__features_pre_processing_type == FeaturePreProcessingType.learning_rate_scaling:
                self.__weights = self.__weights - change * self.__learning_rate.transpose()
            else:
                self.__weights = self.__weights - change * self.__learning_rate
            current_cost = self.__cost_of_training_set()
            if print_cost_while_training:
                print('cost: {0:.2f}'.format(current_cost))
            if current_cost == last_cost:
                cost_not_change_count += 1
            last_cost = current_cost
    
        end_time = time.time()
        print('Used {0:.2f} seconds to train model.'.format(end_time - start_time))
        print('Weights are: {0}'.format(self.__weights))
        try:
            cost_of_testing, error_rate_of_testing = self.__cost_and_error_rate(self.__testing_set_features, self.__testing_set_outputs)
        except RuntimeWarning as e:
            print('Cost for testing samples is too large, can\'t be printed.')
        else:
            print('Cost for {0} testing samples is {1:.2f}'.format(np.size(self.__testing_set_features, axis=0), cost_of_testing))
            print('Error rate for {0} testing samples is ±{1:.2f}%.'.format(np.size(self.__testing_set_features, axis=0), error_rate_of_testing * 100))
        finally:
            print('Training finished.')

    def predict(self, features):
        if self.__features_pre_processing_type == FeaturePreProcessingType.feature_scaling:
            features = self.__scale_features(features)
        
        features = self.__add_x0_column(features)
        return features @ self.__weights.transpose()
    
    #
    # Helper Methods
    #
    
    def __predict_scaled_with_x0_column_features(self, features):
        return features @ self.__weights.transpose()
    
    def __setup_training_and_testing_sets(self):
        num_training_sample, _ = self.__get_training_and_testing_samples_counts()
        self.__training_set_features = self.__data[:num_training_sample, :-1]
        self.__training_set_outputs = self.__data[:num_training_sample, -1]
        self.__testing_set_features = self.__data[num_training_sample:, :-1]
        self.__testing_set_outputs = self.__data[num_training_sample:, -1]
        if self.__features_pre_processing_type != FeaturePreProcessingType.none:
            self.__update_feature_scaling_parameters()
            if self.__features_pre_processing_type == FeaturePreProcessingType.feature_scaling:
                self.__training_set_features = self.__scale_features(self.__training_set_features)
                self.__testing_set_features = self.__scale_features(self.__testing_set_features)
            elif self.__features_pre_processing_type == FeaturePreProcessingType.learning_rate_scaling:
                self.__scale_learning_rate_if_enabled()
        self.__training_set_features = self.__add_x0_column(self.__training_set_features)
        self.__testing_set_features = self.__add_x0_column(self.__testing_set_features)
    
    def __get_training_and_testing_samples_counts(self):
        total_sample_count = np.size(self.__data, axis=0)
        training_set_count = math.ceil((1.0 - self.__test_sample_ratio) * total_sample_count)
        testing_set_count = total_sample_count - training_set_count
        return (training_set_count, testing_set_count)

    def __update_feature_scaling_parameters(self):
        self.__feature_scaling_std = np.std(self.__training_set_features, axis=0)
        self.__feature_scaling_range = np.max(self.__training_set_features, axis=0) - np.min(self.__training_set_features, axis=0)
        
    def __scale_features(self, features):
        return (features - self.__feature_scaling_std) / self.__feature_scaling_range
    
    def __add_x0_column(self, A):
        try:
            return np.insert(A, obj=0, values=1, axis=1)
        except IndexError:
            return np.insert(A, obj=0, values=1)
    
    def __setup_weights(self):
        self.__weights = np.zeros(np.size(self.__training_set_features, axis=1))
        
    def __cost_of_training_set(self):
        result, _ = self.__cost_and_error_rate(self.__training_set_features, self.__training_set_outputs)
        return result
    
    def __cost_of_testing_set(self):
        result, _ = self.__cost_and_error_rate(self.__testing_set_features, self.__testing_set_outputs)
        return result
    
    def __cost_and_error_rate(self, features, outputs):
        predictions = self.__predict_scaled_with_x0_column_features(features)
        diff = np.array(outputs - predictions)
        diff_squared = np.power(diff, 2)
        result_cost = np.average(diff_squared) / 2.0
        result_error_rate = np.average(np.abs(diff) / predictions)
        return (result_cost, result_error_rate)
    
    def __derivative_of_cost(self):
        predictions = self.__predict_scaled_with_x0_column_features(self.__training_set_features)
        diff = predictions - self.__training_set_outputs
        features_scaled_with_diff = (self.__training_set_features.transpose() * diff).transpose()
        return np.average(features_scaled_with_diff, axis=0)

    def __scale_learning_rate_if_enabled(self):
        current_flat_rate = self.__learning_rate
        if self.__features_pre_processing_type == FeaturePreProcessingType.learning_rate_scaling:
            self.__learning_rate *= self.__feature_scaling_std
            self.__learning_rate = np.insert(self.__learning_rate, obj=0, values=current_flat_rate, axis=1)

df = pd.read_csv('housing/housing.data', header=None, delim_whitespace=True)
data = df.as_matrix()

trainer = Trainer(data,
                test_sample_ratio=0.05,
                learning_rate=0.01,
                features_pre_processing_type=FeaturePreProcessingType.feature_scaling)

trainer.train(print_cost_while_training=False)
