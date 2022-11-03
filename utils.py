import xgboost as xgb
from typing import Tuple, Dict
import numpy as np
import itertools


def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.
        :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
        '''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'RMSLE', float(np.sqrt(np.sum(elements) / len(y)))


def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)


def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))


def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    lst = []
    for instance in itertools.product(*vals):
        lst.append(dict(zip(keys, instance)))
    return lst


def xgb_cross_validation(param_lists: Dict, dtrain: xgb.DMatrix, num_boost_round=100, nfold=10, metric=rmsle, early_stopping_rounds=10):
    min_rmsle = float("Inf")
    best_params = None
    for params in product_dict(**param_lists):
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=100,
            seed=42,
            nfold=5,
            custom_metric=rmsle,
            early_stopping_rounds=10)

        # Update best RMSLE
        mean_rmsle = cv_results['test-RMSLE-mean'].min()
        boost_rounds = cv_results['test-RMSLE-mean'].argmin()
        print(params)
        print("\tRMSLE {} for {} rounds \t{}".format(mean_rmsle, boost_rounds, '(New best)' if mean_rmsle < min_rmsle else ''))
        if mean_rmsle < min_rmsle:
            min_rmsle = mean_rmsle
            best_params = params

    print(f"Best params: {best_params}")
    return