import xgboost as xgb
from typing import List, Tuple, Dict
import numpy as np
import itertools
import pandas as pd

from pyproj import Geod
from shapely.geometry import Point, LineString
from sklearn.preprocessing import LabelEncoder


def rmsle_xgb(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
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


def meter_distance(lat1, lon1, lat2, lon2):
    line_string = LineString([Point(lon1, lat1), Point(lon2, lat2)])
    geod = Geod(ellps="WGS84")
    return geod.geometry_length(line_string)


def add_city_centre_dist(X: pd.DataFrame):
    old_shape = X.shape

    city_centres = X.groupby(['municipality_name'])[['lat', 'lon']].apply(lambda x: x.sum() / (x.count()))[['lat', 'lon']]
    X = X.merge(city_centres, on=['municipality_name'], how='left', suffixes=(None, '_center'))
    assert X.shape[0] == old_shape[0]

    X.fillna(value={'lat_center': X.lat, 'lon_center': X.lon}, inplace=True)

    X['dist_to_center'] = X.apply(lambda row: meter_distance(row.lat, row.lon, row.lat_center, row.lon_center), axis=1)
    assert X.shape[0] == old_shape[0]

    return X


def group_ages(age: pd.DataFrame, age_ranges: List[Tuple[int, int]]):
    age_new = age[['grunnkrets_id', 'year']].drop_duplicates(subset=['grunnkrets_id'], keep='last')

    for rng in age_ranges:
        cols = [f'age_{age}' for age in range(rng[0], rng[1] + 1)]
        rng_sum = age[cols].sum(axis=1).astype('int32')
        age_new[f'age_{rng[0]}_{rng[-1]}'] = rng_sum

    age = age.drop_duplicates(subset='grunnkrets_id').drop(columns=['year', *(f'age_{age}' for age in range(0, 91))], axis=1)
    age = age.merge(age_new.drop(columns=['year']), on='grunnkrets_id')

    return age
