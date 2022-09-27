import xgboost as xgb
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

train = pd.read_csv('data/stores_train.csv')
test = pd.read_csv('data/stores_test.csv')

spatial = pd.read_csv('data/grunnkrets_norway_stripped.csv')
age = pd.read_csv('data/grunnkrets_age_distribution.csv')
income = pd.read_csv('data/grunnkrets_income_households.csv')
households = pd.read_csv('data/grunnkrets_households_num_persons.csv')
submission = pd.read_csv('data/sample_submission.csv')
plaace = pd.read_csv('data/plaace_hierarchy.csv')
busstops = pd.read_csv('data/busstops_norway.csv')

model_to_load = "modeling/0002.model"
features = ['store_id', 'revenue', 'year', 'store_name', 'mall_name', 'chain_name', 'address', 'lat', 'lon', 'grunnkrets_id',
            'plaace_hierarchy_id']

print(train.info())
print(busstops.info())

print(busstops.head())


def plot_data(df, store_id):
    df_ = df[['store_id', 'revenue']]
    fig, axs = plt.subplots(2, 1, sharex=True)
    sns.lineplot(df_['store_id'], df_['revenue_diff'], ax=axs[0], color='r')
    # sns.lineplot(df_['revenue'], df_['lon'], ax=axs[1], color='b')
    plt.xticks(rotation=90)
    plt.show()


def generate_features(df, spatial_features, age_features, income_features,
                      households_features, plaace_features, busstops_features):
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

    # attempt to difference the revenue to see if it helps
    df['revenue_diff'] = df['revenue'].diff()

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


# TRAIN BUFFER
if os.path.exists('modeling/train.buffer'):
    print("\ntrain.buffer already exists. Attempting to load...")
    dtrain = xgb.DMatrix('modeling/train.buffer')
    print("--> train.buffer loaded.")

else:
    print("\ntrain.buffer doesn't exist. Attempting to create it...")
    # train = generate_features(train, sku_features, id_map)
    train = generate_features(train, spatial, age, income, households, plaace, busstops)
    train_y = train['revenue']
    del (train['revenue'])
    train_features = train

    dtrain = xgb.DMatrix(data=train_features, label=train_y, enable_categorical=True)
    dtrain.save_binary('modeling/train.buffer')
    print("--> train.buffer created and saved.")
# TEST BUFFER
if os.path.exists('modeling/test.buffer'):
    print("\ntest.buffer already exists. Attempting to load...")
    dtest = xgb.DMatrix('modeling/test.buffer')
    print("--> test.buffer loaded.")

else:
    print("\ntest.buffer doesn't exist. Attempting to create it...")
    test['revenue'] = 0
    test = generate_features(test, spatial, age, income, households, plaace, busstops)
    test_label = test['revenue']
    del (test['revenue'])
    test_features = test

    dtest = xgb.DMatrix(data=test_features, label=test_label, enable_categorical=True)
    # dtest = xgb.DMatrix(test_features)
    dtest.save_binary('modeling/test.buffer')
    print("--> test.buffer created and saved.")

# check if there already exists a model.
if os.path.exists(model_to_load):
    print("\nModel found, attempting to load.")
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(model_to_load)  # load data
    print("--> model successfully loaded.")
else:
    print("\nNo model found. Attempt at creating a new one will now start:")
    print("Attempting to initialize parameters for training...")
    param = {'max_depth': 10, 'eta': 0.1, 'objective': 'reg:squarederror'}
    num_round = 1000
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    print("--> parameters for training initialized.")

    print("Attempting to start training...")
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    print("--> model trained.")

    print("Attempting to save model...")
    bst.save_model(model_to_load)
    print("--> model saved.")

plot_data(generate_features(train, spatial, age, income, households, plaace, busstops), 0)

print("\nAttempting to start prediction...")
ypred = bst.predict(dtest, ntree_limit=bst.best_iteration)
print("--> Prediction finished.")

print("\nAttempting to save prediction...")
submission['predicted'] = np.array(ypred)
sub_features = features[1:len(features)]
submission.to_csv('submissions/'+','.join(sub_features)+'.csv', index=False)
print("--> prediction saved with features as name in submission folder.")


