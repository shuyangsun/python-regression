import enum


class RegressionAlgorithm(enum.Enum):
    unspecified = 0
    gradient_descent = 1
    normal_equation = 2
