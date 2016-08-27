import numpy as np
import regression.regpkg_shuyang.reg_predictor.predictor_super as predictor_super
import regression.regpkg_shuyang.data_processor as data_processor


class RegressionPredictorLinear(predictor_super.RegressionPredictor):
    def predict(self, data):
        normalized_feature = self._feature_normalizer.normalize_new_feature(data, False)
        normalized_feature = data_processor.DataProcessor.add_x0_column(normalized_feature)
        return np.array(self._weights @ normalized_feature.transpose())
