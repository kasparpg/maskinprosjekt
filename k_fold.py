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


def _rmsle(y, y_pred):
    y_pred[y_pred < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y_pred) - np.log1p(y), 2)
    return 'RMSLE', float(np.sqrt(np.sum(elements) / len(y)))


rmsle_scorer = make_scorer(lambda y, y_true: _rmsle(y, y_true)[1], greater_is_better=False)


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


def random_k_fold(X, y, model=None, params=None, k=5, n_iter=20, verbose=10, n_jobs=-1):
    """Does k-fold cross validation."""
    # Parameter grid for XGBoost
    params = {"colsample_bytree": uniform(0.4, 0.5),
              "gamma": uniform(0, 0.5),
              "learning_rate": uniform(0.01, 0.3),  # default 0.1
              "max_depth": randint(3, 9),  # default 3
              "n_estimators": randint(150, 350),  # default 100
              "subsample": uniform(0.6, 0.4),
              'objective': ['reg:squaredlogerror'],
              'eval_metric': ['rmsle'],
              'min_child_weight': randint(1, 5),
              'max_depth': randint(5, 9)} if params is None else params

    kfold = KFold(n_splits=k, shuffle=True)

    model = XGBRegressor() if model is None else model

    rand_search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter,
                                   scoring=rmsle_scorer, verbose=verbose,
                                   cv=kfold.split(X, y), n_jobs=n_jobs)

    start = time.time()
    model = rand_search.fit(X, y)
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
        n_iters = params.pop('num_boost_rounds')
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=300,
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
