import abc
import regression.regpkg_shuyang.reg_alg as reg_alg


class RegressionSetup(metaclass=abc.ABCMeta):
    def __init__(self,
                 data,
                 test_sample_ratio=0.05,
                 learning_rate=0.01,
                 regularization_lambda=0.0,
                 regression_algorithm=reg_alg.RegressionAlgorithm.unspecified):
        if data is None:
            raise ValueError('Cannot initialize regression setup, no data.')
        if not 0 <= test_sample_ratio < 1:
            raise ValueError('Cannot initialize regression setup, invaild test sample ratio.')

        self.data = data
        self.test_sample_ratio = test_sample_ratio
        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda
        self.regression_algorithm = regression_algorithm

    @abc.abstractproperty
    def regression_type(self):
        pass
