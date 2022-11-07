import pandas
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
import joblib
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import make_scorer
import time

from xgboost import plot_importance
import warnings

from shapely import wkt
from sklearn.model_selection import train_test_split
from utils import squared_log, rmsle
from xgboost import plot_importance, to_graphviz

cat_features = list(X_train.select_dtypes(include=['category']).columns)

pool_train = Pool(X_train, y_train, cat_features=cat_features)
pool_test = Pool(X_test, cat_features=cat_features)

parameters = {'depth': randint(2, 20),
              'learning_rate': uniform(0.01, 0.4),
              'iterations': randint(10, 1000)
              }


model = CatBoostRegressor(verbose=False, cat_features=cat_features, eval_metric="RMSE")

random_k_fold(X_train, y_train, model, parameters, cat_features, verbose=10)
y_pred = model.predict(pool_test)

submission['predicted'] = np.array(y_pred)
submission.to_csv('submissions/submission.csv', index=False)
