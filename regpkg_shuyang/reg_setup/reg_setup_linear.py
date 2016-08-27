import regression.regpkg_shuyang.reg_setup.reg_setup_super as reg_setup_super
import regression.regpkg_shuyang.reg_type as reg_type


class RegressionSetupLinear(reg_setup_super.RegressionSetup):
    # Override (abstract)
    @property
    def regression_type(self):
        return reg_type.RegressionType.linear
