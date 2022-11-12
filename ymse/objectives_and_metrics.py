import numpy as np
import xgboost as xgb

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error
from typing import Tuple


def rmsle(y_true, y_pred):
    y_pred[y_pred < 0] = 0 + 1e-6
    y_true[y_true < 0] = 0 + 1e-6
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


rmsle_scorer = make_scorer(lambda y, y_true: rmsle(y, y_true), greater_is_better=False)


def rmsle_xgb(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.

        :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
        '''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'RMSLE', float(np.sqrt(np.sum(elements) / len(y)))


class RmseObjective:
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            der1 = targets[index] - approxes[index]
            der2 = -1

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result


class RmsleObjective:
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            der1 = np.log1p(targets[index]) - np.log1p(approxes[index])
            der2 = -1

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result


class RmsleMetric:
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((np.log1p(approx[i]) - np.log1p(target[i]))**2)

        return error_sum, weight_sum


class LogTargetsRmsleMetric:
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approxes, target = np.expm1(approxes), np.expm1(target)

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            # error_sum += w * ((np.expm1(approx[i]) - np.expm1(target[i]))**2)
            # error_sum += w * ((approx[i] - target[i])**2)
            error_sum += w * ((np.log1p(approx[i]) - np.log1p(target[i]))**2)

        return error_sum, weight_sum