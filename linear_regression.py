import numpy as np
import math
import sys
import time

class Trainer:
    """
    This class is used to perform linear regression.
    """
    def __init__(self, data, test_sample_ratio, learning_rate=0.01, is_feature_scaling_enabled=True):
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
        self.__is_feature_scaling_enabled = is_feature_scaling_enabled
        
        self.__setup_training_and_testing_sets()
        self.__setup_weights()

    def train(self):
        start_time = time.time()
        last_cost = self.__cost()
        cost_not_change_count = 0
        while cost_not_change_count <= 3:
            predictions = self.__predict_scaled_with_x0_column_features(self.__training_set_features)
            diff = predictions - self.__training_set_outputs
            features_scaled_with_diff = (self.__training_set_features.transpose() @ diff).transpose()
            change = self.__derivative_of_cost()
            self.__weights = self.__weights - self.__learning_rate * change
            current_cost = self.__cost()
            if current_cost == last_cost:
                cost_not_change_count += 1
            last_cost = current_cost
    
        end_time = time.time()
        print('Finished trainging, used {0:.2f} seconds.'.format(end_time - start_time))
        print('Weights are: {0}'.format(self.__weights))
        try:
            print('Error rate is {0:.10f}%.'.format(self.__get_error_rate_from_testing_set() * 100))
        except ValueError as e:
            print('There is no testing data.')
        
    def predict(self, features):
        if self.__is_feature_scaling_enabled:
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
        if self.__is_feature_scaling_enabled:
            self.__update_feature_scaling_parameters()
            self.__training_set_features = self.__scale_features(self.__training_set_features)
        self.__training_set_features = self.__add_x0_column(self.__training_set_features)
    
    def __get_training_and_testing_samples_counts(self):
        total_sample_count = np.size(self.__data, axis=0)
        training_set_count = math.ceil((1.0 - self.__test_sample_ratio) * total_sample_count)
        testing_set_count = total_sample_count - training_set_count
        return (training_set_count, testing_set_count)

    def __update_feature_scaling_parameters(self):
        self.__feature_scalling_avg = np.average(self.__training_set_features, axis=0)
        self.__feature_scalling_range = np.max(self.__training_set_features, axis=0) - np.min(self.__training_set_features, axis=0)
        
    def __scale_features(self, features):
        return (features - self.__feature_scalling_avg) / self.__feature_scalling_range
    
    def __add_x0_column(self, A):
        try:
            return np.insert(A, obj=0, values=1, axis=1)
        except IndexError:
            return np.insert(A, obj=0, values=1)
    
    def __setup_weights(self):
        self.__weights = np.zeros(np.size(self.__training_set_features, axis=1))
        
    def __cost(self):
        predictions = np.sum(self.__training_set_features @ self.__weights.transpose())
        diff = self.__training_set_outputs - predictions
        diff_squared = diff * diff.transpose()
        result = np.sum(diff_squared) * 0.5 / np.size(self.__training_set_features, axis=0)
        return result
    
    def __derivative_of_cost(self):
        predictions = self.__predict_scaled_with_x0_column_features(self.__training_set_features)
        diff = predictions - self.__training_set_outputs
        features_scaled_with_diff = (self.__training_set_features.transpose() @ diff).transpose()
        return np.average(features_scaled_with_diff, axis=0)
    
    def __get_error_rate_from_testing_set(self):
        if not self.__testing_set_features:
            raise ValueError('Cannot get error rate, does not have a testing set data.')
        predictions = self.predict(self.__testing_set_features)
        errors = predictions - self.__testing_set_outputs
        error_rates = abs(errors) / self.__testing_set_outputs
        return np.average(error_rates)

data = np.matrix('1 20; 3 40; 5 60; 0 10; 10 110; -1 0; 9.5 105; 3.5 45; -10 -90; 10 110')
trainer = Trainer(data, test_sample_ratio=0.1, learning_rate=0.001, is_feature_scaling_enabled=False)
trainer.train()

print(trainer.predict([3]))


            
        
    
