import regression.regpkg_shuyang.reg_setup.reg_setup_super as reg_setup_super
import regression.regpkg_shuyang.reg_type as reg_type
import regression.regpkg_shuyang.reg_alg as reg_alg


class RegressionSetupLogistic(reg_setup_super.RegressionSetup):
    def __init__(self,
                 data,
                 test_sample_ratio=0.05,
                 learning_rate=0.01,
                 regularization_lambda=0.0,
                 regression_algorithm=reg_alg.RegressionAlgorithm.unspecified,
                 output_case_sensitive=True):
        super().__init__(data, test_sample_ratio, learning_rate, regularization_lambda, regression_algorithm)
        self.output_case_sensitive = output_case_sensitive

    # Override (abstract)
    @property
    def regression_type(self):
        return reg_type.RegressionType.logistic
