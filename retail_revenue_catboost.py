import pandas
from catboost import CatBoostRegressor, Pool
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



from xgboost import plot_importance
import warnings

from shapely import wkt
from sklearn.model_selection import train_test_split
from utils import squared_log, rmsle
from xgboost import plot_importance, to_graphviz

# warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

train = pd.read_csv('../data/stores_train.csv')
test = pd.read_csv('../data/stores_test.csv')
extra = pd.read_csv('../data/stores_extra.csv')

spatial = pd.read_csv('../data/grunnkrets_norway_stripped.csv')
age = pd.read_csv('../data/grunnkrets_age_distribution.csv')
income = pd.read_csv('../data/grunnkrets_income_households.csv')
households = pd.read_csv('../data/grunnkrets_households_num_persons.csv')
submission = pd.read_csv('../data/sample_submission.csv')
plaace = pd.read_csv('../data/plaace_hierarchy.csv')
busstops = pd.read_csv('../data/busstops_norway.csv')

model_to_load = "modeling/0002.model"

def plot_data(df):
    df_ = df[['all_households', 'revenue']]
    fig, axs = plt.subplots(2, 1, sharex=True)
    #sns.lineplot(df_['grunnkrets_id'], df_['age_children'], ax=axs[0], color='r')
    #sns.lineplot(df_['grunnkrets_id'], df_['age_youth'], ax=axs[1], color='b')
    plt.xticks(rotation=90)
    plt.scatter(df_['all_households'], df_['revenue'])
    plt.show()
    """
    for i in np.unique(df['clusters']):
        plt.scatter(df[['clusters'] == i, 0], df[['clusters'] == i, 1], label=i)
    plt.legend()
    plt.show()
    """


def generate_features(df: pd.DataFrame):
    features = ['year', 'store_name', 'mall_name', 'chain_name', 'address', 'lat', 'lon',
                'plaace_hierarchy_id', 'grunnkrets_id']
    df = df[features]

    # # # # # # # # # # # # # #
    # # FEATURE ENGINEERING # #

    """ THESE REDUCE SCORE DRASTICALLY
    df['address'] = df['address'].fillna("NO ADDRESS")
    df['mall_name'] = df['mall_name'].fillna("No Mall")
    df['chain_name'] = df['chain_name'].fillna("No Chain")
    """
    """
    df['store_name_address'] = (df['store_name'].astype(str) + df['address'].astype(str)).astype('category')
    df['store_name_lat'] = (df['store_name'].astype(str) + df['lat'].astype(str)).astype('category')
    df['store_name_lon'] = (df['store_name'].astype(str) + df['lon'].astype(str)).astype('category')
    """
    # joblib.dump(kmeans, "kmeans/kmeans"+str(rounds)+".joblib")

    kmeans = joblib.load("kmeans_train/kmeans189.joblib")
    # kmeans = KMeans(n_clusters=110, random_state=0).fit(np.column_stack((df['lat'], df['lon'])))
    # joblib.dump(kmeans, "kmeans_train/kmeans" + str(rounds) + ".joblib")

    df['clusters'] = kmeans.predict(np.column_stack((df['lat'], df['lon'])))
    # one hot encoding
    """
    y = pd.get_dummies(df.chain_name, prefix='Chain')
    df.append(y)
    """

    # # # # # # # # # # # # #
    # # FEATURE SELECTION # #

    df['store_name'] = df['store_name'].fillna(value="None").astype('category')
    """
    if os.path.exists('modeling/word2vec_model'):
        word2vec_model = gensim.models.Word2Vec.load('modeling/word2vec_model')
    else:
        word2vec_model = Word2Vec(sentences=df['store_name'], min_count=1, vector_size=200, workers=4)
        word2vec_model.save('modeling/word2vec_model')
    """
    """
    card_docs = [TaggedDocument(doc.split(' '), [i])
                 for i, doc in enumerate(df.store_name)]
    
    doc2vec_model = Doc2Vec(min_count=1, vector_size=5, workers=4, epochs=40)
    doc2vec_model.build_vocab(card_docs)
    doc2vec_model.train(card_docs, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    card2vec = [doc2vec_model.infer_vector((df['store_name'][i].split(' ')))
                for i in range(0, len(df['store_name']))]
    dtv = np.array(card2vec)
    df['store_name_embedded1'] = dtv[:, 0]
    df['store_name_embedded2'] = dtv[:, 1]
    df['store_name_embedded3'] = dtv[:, 2]
    df['store_name_embedded4'] = dtv[:, 3]
    df['store_name_embedded5'] = dtv[:, 4]
    df = df.drop(columns=['store_name'])
    """

    """
    y = pd.get_dummies(df.store_name, prefix='store')
    df.append(y)
    """

    # df['store_id'] = df['store_id'].astype('category')
    df['address'] = df['address'].fillna(value="None").astype('category')
    df['mall_name'] = df['mall_name'].fillna(value="None").astype('category')
    df['chain_name'] = df['chain_name'].fillna(value="None").astype('category')
    # df = df.drop(columns=["chain_name"])  # drop as we have one hot encoded it
    df['plaace_hierarchy_id'] = df['plaace_hierarchy_id'].fillna(value="None").astype('category')

    spatial.drop_duplicates(subset=['grunnkrets_id'])  # remove duplicates from spatial data
    df = pd.merge(df, spatial.drop_duplicates(subset=['grunnkrets_id']), how='left')  # merge with the spatial data.
    df['grunnkrets_name'] = df['grunnkrets_name'].fillna(value="None").astype('category')
    df['district_name'] = df['district_name'].fillna(value="None").astype('category')
    df['municipality_name'] = df['municipality_name'].fillna(value="None").astype('category')
    df['geometry'] = df['geometry'].fillna(value="None").astype('category')

    """
    age.drop_duplicates(subset=['grunnkrets_id'])  # remove duplicates from age data
    df = pd.merge(df, age.drop_duplicates(subset=['grunnkrets_id']), how='left')  # merge with the age data.
    age_children = ['age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'age_9',
                    'age_10', 'age_11', 'age_12', 'age_13', 'age_14']
    age_youth = ['age_15', 'age_16', 'age_17', 'age_18', 'age_19', 'age_20', 'age_21', 'age_22', 'age_23', 'age_24']
    age_adults = ['age_25', 'age_26', 'age_27', 'age_28', 'age_29', 'age_30', 'age_31', 'age_32', 'age_33', 'age_34',
                  'age_35', 'age_36', 'age_37', 'age_38', 'age_39', 'age_40', 'age_41', 'age_42', 'age_43', 'age_44',
                  'age_45', 'age_46', 'age_47', 'age_48', 'age_49', 'age_50', 'age_51', 'age_52', 'age_53', 'age_54',
                  'age_55', 'age_56', 'age_57', 'age_58', 'age_59', 'age_60', 'age_61', 'age_62', 'age_63', 'age_64']
    age_senior = ['age_65', 'age_66', 'age_67', 'age_68', 'age_69', 'age_70', 'age_71', 'age_72',
                  'age_73', 'age_74', 'age_75', 'age_76', 'age_77', 'age_78', 'age_79', 'age_80', 'age_81',
                  'age_82', 'age_83', 'age_84', 'age_85', 'age_86', 'age_87', 'age_88', 'age_89', 'age_90']
    df['age_children'] = df[age_children].sum(axis=1)
    df['age_youth'] = df[age_youth].sum(axis=1)
    df['age_adults'] = df[age_adults].sum(axis=1)
    df['age_senior'] = df[age_senior].sum(axis=1)
    df.drop(age_children, axis=1, inplace=True)
    df.drop(age_youth, axis=1, inplace=True)
    df.drop(age_adults, axis=1, inplace=True)
    df.drop(age_senior, axis=1, inplace=True)
    """

    """ REDUCES SCORE
    df['age0-16'] = df['age0-16'].replace(0, df['age0-16'].mean())
    df['age17-45'] = df['age17-45'].replace(0, df['age17-45'].mean())
    df['age46-90'] = df['age46-90'].replace(0, df['age46-90'].mean())
    """

    income.drop_duplicates(subset=['grunnkrets_id'])  # remove duplicates from income data
    df = pd.merge(df, income.drop_duplicates(subset=['grunnkrets_id']), how='left')  # merge with the income data.

    # df['all_households'] = df['all_households'].replace(0, df['all_households'].mean()) REDUCES SCORE, BUT WHY?

    households.drop_duplicates(subset=['grunnkrets_id'])  # remove duplicates from households data
    df = pd.merge(df, households.drop_duplicates(subset=['grunnkrets_id']), how='left')  # merge with the households data.


    plaace.drop_duplicates(subset=['plaace_hierarchy_id'])  # remove duplicates from plaace data
    df = pd.merge(df, plaace.drop_duplicates(subset=['plaace_hierarchy_id']), how='left')  # merge with the plaace data.
    df['plaace_hierarchy_id'] = df['plaace_hierarchy_id'].fillna(value="None").astype('category')
    df['sales_channel_name'] = df['sales_channel_name'].fillna(value="None").astype('category')
    df['lv1_desc'] = df['lv1_desc'].fillna(value="None").astype('category')
    df['lv2_desc'] = df['lv2_desc'].fillna(value="None").astype('category')
    df['lv3'] = df['lv3'].fillna(value="None").astype('category')
    df['lv3_desc'] = df['lv3_desc'].fillna(value="None").astype('category')
    df['lv4'] = df['lv4'].fillna(value="None").astype('category')
    df['lv4_desc'] = df['lv4_desc'].fillna(value="None").astype('category')

    """ NO DIFFERENCE
    y = pd.get_dummies(df.lv1_desc, prefix='LV1')
    df.append(y)
    df.drop(columns=['lv1_desc'])  # drop as we have one hot encoded it
    """


    busstops.drop_duplicates(subset=['geometry'])  # remove duplicates from busstops data
    df = pd.merge(df, busstops.drop_duplicates(subset=['geometry']), how='left')  # merge with the busstops data

    df['busstop_id'] = df['busstop_id'].fillna(value="None").astype('category')
    df['stopplace_type'] = df['stopplace_type'].fillna(value="None").astype('category')
    df['importance_level'] = df['importance_level'].fillna(value="None").astype('category')
    df['side_placement'] = df['side_placement'].fillna(value="None").astype('category')
    df['geometry'] = df['geometry'].fillna(value="None").astype('category')


    # df.drop(columns=["store_id"])
    df = df.drop(columns=['grunnkrets_id', 'plaace_hierarchy_id'])

    return df

def random_k_fold(X, y, model=None, params=None, k=5, n_iter=20, verbose=10, n_jobs=-1):
    # Parameter grid for XGBoost
    params = {"colsample_bytree": uniform(0.3, 0.7),
              "gamma": uniform(0, 0.5),
              "learning_rate": uniform(0.003, 0.3),  # default 0.1
              "max_depth": randint(2, 6),  # default 3
              "n_estimators": randint(100, 400),  # default 100
              "subsample": uniform(0.6, 0.4),
              'objective': ['reg:squaredlogerror'],
              'eval_metric': [_rmsle],
              'min_child_weight': randint(3, 8),
              'max_depth': randint(5, 10)} if params is None else params

    kfold = KFold(n_splits=k, shuffle=True)

    model = XGBRegressor() if model is None else model
    randm_src = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter,
                                   scoring=rmsle_scorer, verbose=verbose,
                                   cv=kfold.split(X, y), n_jobs=n_jobs)

    start = time.time()
    model = randm_src.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter))

    return model

label_name = 'revenue'
X = train.drop(columns=[label_name])
y = train[label_name]

# plot_data(generate_features(train))

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.8)
X_train, X_val = generate_features(X_train), generate_features(X_val)
X_test = generate_features(test)

pool_train = Pool(X_train, y_train, cat_features=list(X_train.select_dtypes(include=['category']).columns))

pool_test = Pool(X_test, cat_features=list(X_test.select_dtypes(include=['category']).columns))

model = CatBoostRegressor(iterations=2,
                          learning_rate=1,
                          depth=2)

model.fit(pool_train)

y_pred = model.predict(pool_test)

submission['predicted'] = np.array(y_pred)
submission.to_csv('submissions/submission.csv', index=False)