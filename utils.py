import xgboost as xgb
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np

from pyproj import Geod
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from tqdm import tqdm

tqdm.pandas()


def to_categorical(df: pd.DataFrame):
    for cat_col in df.select_dtypes(include=[object]).columns:
        df[cat_col] = df[cat_col].astype('category')

    return df


def object_encoder(df: pd.DataFrame):
    enc = OrdinalEncoder()
    obj_cols = df.select_dtypes(include=[object]).columns
    df[obj_cols] = enc.fit_transform(df[obj_cols])
    return df


def nan_to_string(df: pd.DataFrame):
    nan = '#N/A'
    cols = df[df.columns[df.isna().any()]].columns
    df[cols] = df[cols].fillna(nan)
    return df


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
        rng_sum = age[cols].sum(axis=1).astype(int)
        age_new[f'age_{rng[0]}_{rng[-1]}'] = rng_sum

    age = age.drop_duplicates(subset='grunnkrets_id').drop(columns=['year', *(f'age_{age}' for age in range(0, 91))], axis=1)
    age = age.merge(age_new.drop(columns=['year']), on='grunnkrets_id')

    return age


def only_2016_data(df: pd.DataFrame):
    df = df.sort_values(by='year', ascending=False)
    df = df.drop_duplicates(subset='grunnkrets_id', keep='first')
    return df


def clean_out_nan_heavy_rows(df: pd.DataFrame, age, age_ranges, spatial_2016, income_2016, households_2016):
    """Cleans out rows that have no match in the age, spatial, income or household datasets."""

    df2 = df.merge(group_ages(age, age_ranges), on='grunnkrets_id', how='left')
    df2 = df2.merge(spatial_2016.drop(columns=['year']), on='grunnkrets_id', how='left')
    df2 = df2.merge(income_2016.drop(columns=['year']), on='grunnkrets_id', how='left')
    df2 = df2.merge(households_2016.drop(columns=['year']), on='grunnkrets_id', how='left')

    df_cleaned = df2[
        ~(df2.age_0_19.isna() | df2.couple_children_0_to_5_years.isna() | df2.grunnkrets_name.isna() | df2.income_all_households.isna())
    ]

    print(f'Cleaned out {len(df) - len(df_cleaned)} out of {len(df)} rows.')

    return df_cleaned


def add_spatial_clusters(df: pd.DataFrame):
    clusters = DBSCAN(eps=0.145, min_samples=100)
    # clusters = DBSCAN(eps=0.12, min_samples=30)
    cl = clusters.fit_predict(df[['lat', 'lon']].to_numpy())
    cl_counts = dict(zip(*np.unique(cl, return_counts=True)))

    print(len(set(cl)), 'clusters created')
    print('Cluster counts:', cl_counts)

    df['cluster_id'] = cl
    df['cluster_member_count'] = df.apply(lambda row: cl_counts[row.cluster_id], axis=1)

    X_no_outliers = df[df.cluster_id != -1]
    cluster_centroids = X_no_outliers.groupby('cluster_id')[['lat', 'lon']].mean()

    def closest_centroid(lat, lon):
        dist_series = cluster_centroids.apply(lambda row: meter_distance(lat, lon, row.lat, row.lon), axis=1)
        return dist_series.min()

    print('Calculating distance to closest cluster for each data point...')
    df['closest_cluster_centroid_dist'] = df.progress_apply(lambda row: closest_centroid(row.lat, row.lon), axis=1)
    
    return df

