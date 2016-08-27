import abc


class RegressionPredictor(metaclass=abc.ABCMeta):
    def __init__(self, weights, feature_normalizer):
        self._weights = weights
        self._feature_normalizer = feature_normalizer

    @abc.abstractmethod
    def predict(self, data):
        pass
