import numpy as np
import regression.regpkg_shuyang.reg_predictor.predictor_super as predictor_super
import regression.regpkg_shuyang.data_processor as data_processor


class RegressionPredictorLogistic(predictor_super.RegressionPredictor):
    # Override
    def __init__(self, weights, feature_normalizer, categories):
        super().__init__(weights, feature_normalizer)
        self.__categories = categories

    # Override (abstract)
    def predict(self, data):
        normalized_feature = self._feature_normalizer.normalize_new_feature(data, False)
        normalized_feature = (data_processor.DataProcessor.add_x0_column(normalized_feature)).astype(np.float64)
        hypothesis = self._weights.astype(np.float64) @ normalized_feature.transpose()
        if np.size(self.__categories) <= 2:
            return np.array([self.__categories[0] if p >= 0.5 else self.__categories[1] for p in hypothesis])
        else:
            max_ind = np.argmax(hypothesis, axis=0)
            return np.array([self.__categories[ind] for ind in max_ind])
