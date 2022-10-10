from typing import List
import xgboost as xgb
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_features(df: pd.DataFrame, spatial_features, age_features, income_features,
                      households_features, plaace_features, busstops_features):
    features = ['store_id', 'revenue', 'year', 'store_name', 'mall_name', 'chain_name', 'address', 'lat', 'lon', 'grunnkrets_id',
                'plaace_hierarchy_id']
    df = df[features]
    df['store_name'] = df['store_name'].astype('category')
    df['store_id'] = df['store_id'].astype('category')
    df['address'] = df['address'].astype('category')
    df['chain_name+mall_name'] = (df['chain_name'] + df['mall_name']).astype('category')
    df['mall_name'] = df['mall_name'].astype('category')
    df['chain_name'] = df['chain_name'].astype('category')
    df['plaace_hierarchy_id'] = df['plaace_hierarchy_id'].astype('category')

    # attempt to difference the lat and lon values, as they seem to be somewhat trending negatively.
    df['lon'] = df['lon'].diff()
    df['lat'] = df['lat'].diff()

    # # attempt to difference the revenue to see if it helps
    # df['revenue_diff'] = df['revenue'].diff()

    # remove duplicates and merge with the spatial data.
    spatial_features.drop_duplicates(subset=['grunnkrets_id'])
    df = pd.merge(df, spatial_features.drop_duplicates(subset=['grunnkrets_id']), how='left')
    df['grunnkrets_name'] = df['grunnkrets_name'].astype('category')
    df['district_name'] = df['district_name'].astype('category')
    df['municipality_name'] = df['municipality_name'].astype('category')
    df['geometry'] = df['geometry'].astype('category')

    age_features.drop_duplicates(subset=['grunnkrets_id'])
    df = pd.merge(df, age_features.drop_duplicates(subset=['grunnkrets_id']), how='left')

    income_features.drop_duplicates(subset=['grunnkrets_id'])
    df = pd.merge(df, income_features.drop_duplicates(subset=['grunnkrets_id']), how='left')

    households_features.drop_duplicates(subset=['grunnkrets_id'])
    df = pd.merge(df, households_features.drop_duplicates(subset=['grunnkrets_id']), how='left')

    plaace_features.drop_duplicates(subset=['plaace_hierarchy_id'])
    df = pd.merge(df, plaace_features.drop_duplicates(subset=['plaace_hierarchy_id']), how='left')
    df['plaace_hierarchy_id'] = df['plaace_hierarchy_id'].astype('category')
    df['sales_channel_name'] = df['sales_channel_name'].astype('category')
    df['lv1_desc'] = df['lv1_desc'].astype('category')
    df['lv2_desc'] = df['lv2_desc'].astype('category')
    df['lv3'] = df['lv3'].astype('category')
    df['lv3_desc'] = df['lv3_desc'].astype('category')
    df['lv4'] = df['lv4'].astype('category')
    df['lv4_desc'] = df['lv4_desc'].astype('category')

    busstops_features.drop_duplicates(subset=['geometry'])
    df = pd.merge(df, busstops_features.drop_duplicates(subset=['geometry']), how='left')
    df['busstop_id'] = df['busstop_id'].astype('category')
    df['stopplace_type'] = df['stopplace_type'].astype('category')
    df['importance_level'] = df['importance_level'].astype('category')
    df['side_placement'] = df['side_placement'].astype('category')
    df['geometry'] = df['geometry'].astype('category')

    return df


def create_buffer(buffer_path: str, features: pd.DataFrame, data: pd.DataFrame, label_name: str = None):
    file_name = buffer_path.split('/')[-1]

    if os.path.exists(buffer_path):
        print(f'\n{file_name} already exists. Attempting to load...')
        dm = xgb.DMatrix(buffer_path)
        print(f'--> {file_name} loaded.')
    else:
        print(f'\n{file_name} doesn\'t exist. Attempting to create it...')

        labels = features[label_name] if label_name in features else None
        features = features.drop(columns=[label_name])

        dm = xgb.DMatrix(data=features, label=labels, enable_categorical=True)
        dm.save_binary(buffer_path)
        print(f'--> {file_name} created and saved.')

    return dm


def create_model(model_path: str):
    if os.path.exists(model_path):
        print("\nModel found, attempting to load.")
        model = xgb.Booster({'nthread': 4})  # init model
        model.load_model(model_path)  # load data
        print("--> model successfully loaded.")
    else:
        print("\nNo model found. Attempt at creating a new one will now start:")
        print("Attempting to initialize parameters for training...")
        param = {'max_depth': 10, 'eta': 0.1, 'objective': 'reg:squarederror'}
        num_round = 1000
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        print("--> parameters for training initialized.")

        print("Attempting to start training...")
        model = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
        print("--> model trained.")

        print("Attempting to save model...")
        model.save_model(model_path)
        print("--> model saved.")

    return model
