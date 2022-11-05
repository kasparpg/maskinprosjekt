import itertools
import xgboost as xgb
import numpy as np
import time

from xgboost import XGBRegressor
from typing import Dict
from utils import rmsle_xgb
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import make_scorer


def _rmsle(y_pred, y_true):
    y_pred[y_pred < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y_pred) - np.log1p(y_true), 2)
    return 'RMSLE', float(np.sqrt(np.sum(elements) / len(y_true)))


def _rmsle_vanilla(y_pred, y_true):
    elements = np.power(np.log1p(y_pred) - np.log1p(y_true), 2)
    return float(np.sqrt(np.sum(elements) / len(y_true)))


_rmsle_scorer = make_scorer(_rmsle_vanilla, greater_is_better=False)


def random_k_fold(X, y, model=None, params=None, k=5, n_iter=20, verbose=10, n_jobs=-1):
    """ Does k-fold cross validation. No output unless n_jobs == 1. """
    # Parameter grid for XGBoost
    params = {"colsample_bytree": uniform(0.4, 0.5),
              "gamma": uniform(0, 0.5),
              "learning_rate": uniform(0.003, 0.3),  # default 0.1
              "max_depth": randint(3, 9),  # default 3
              "n_estimators": randint(100, 350),  # default 100
              "subsample": uniform(0.6, 0.4),
              'objective': ['reg:squaredlogerror'],
              'eval_metric': [_rmsle],
              'min_child_weight': randint(1, 6),
              'max_depth': randint(5, 10)} if params is None else params

    kfold = KFold(n_splits=k, shuffle=True)

    model = XGBRegressor() if model is None else model
    randm_src = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter,
                                   scoring=_rmsle_scorer, verbose=verbose,
                                   cv=kfold.split(X, y), n_jobs=n_jobs)

    start = time.time()
    model = randm_src.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter))

    return model


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    lst = []
    for instance in itertools.product(*vals):
        lst.append(dict(zip(keys, instance)))
    return lst


def xgb_cross_validation(param_lists: Dict, dtrain: xgb.DMatrix, num_boost_round=100, nfold=10, metric=rmsle_xgb, early_stopping_rounds=10):
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
            custom_metric=rmsle_xgb,
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
    return best_params
