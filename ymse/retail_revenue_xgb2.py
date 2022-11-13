import pandas
import xgboost as xgb
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
                'plaace_hierarchy_id', 'grunnkrets_id', 'revenue']
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

    # kmeans = joblib.load("kmeans_train/kmeans"+str(kmeans_round)+".joblib")
    # kmeans = KMeans(n_clusters=110, random_state=0).fit(np.column_stack((df['lat'], df['lon'])))
    # joblib.dump(kmeans, "kmeans_train/kmeans" + str(rounds) + ".joblib")

    # df['clusters'] = kmeans.predict(np.column_stack((df['lat'], df['lon'])))
    # one hot encoding
    """
    y = pd.get_dummies(df.chain_name, prefix='Chain')
    df.append(y)
    """

    # # # # # # # # # # # # #
    # # FEATURE SELECTION # #

    df['store_name'] = df['store_name'].astype('category')
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
    df['address'] = df['address'].astype('category')
    df['mall_name'] = df['mall_name'].astype('category')
    df['chain_name'] = df['chain_name'].astype('category')
    # df = df.drop(columns=["chain_name"])  # drop as we have one hot encoded it
    df['plaace_hierarchy_id'] = df['plaace_hierarchy_id'].astype('category')

    spatial.drop_duplicates(subset=['grunnkrets_id'])  # remove duplicates from spatial data
    df = pd.merge(df, spatial.drop_duplicates(subset=['grunnkrets_id']), how='left')  # merge with the spatial data.
    df['grunnkrets_name'] = df['grunnkrets_name'].astype('category')
    df['district_name'] = df['district_name'].astype('category')
    df['municipality_name'] = df['municipality_name'].astype('category')
    df['geometry'] = df['geometry'].astype('category')

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
    df['plaace_hierarchy_id'] = df['plaace_hierarchy_id'].astype('category')
    df['sales_channel_name'] = df['sales_channel_name'].astype('category')
    df['lv1_desc'] = df['lv1_desc'].astype('category')
    df['lv2_desc'] = df['lv2_desc'].astype('category')
    df['lv3'] = df['lv3'].astype('category')
    df['lv3_desc'] = df['lv3_desc'].astype('category')
    df['lv4'] = df['lv4'].astype('category')
    df['lv4_desc'] = df['lv4_desc'].astype('category')

    """ NO DIFFERENCE
    y = pd.get_dummies(df.lv1_desc, prefix='LV1')
    df.append(y)
    df.drop(columns=['lv1_desc'])  # drop as we have one hot encoded it
    """


    busstops.drop_duplicates(subset=['geometry'])  # remove duplicates from busstops data
    df = pd.merge(df, busstops.drop_duplicates(subset=['geometry']), how='left')  # merge with the busstops data

    df['busstop_id'] = df['busstop_id'].astype('category')
    df['stopplace_type'] = df['stopplace_type'].astype('category')
    df['importance_level'] = df['importance_level'].astype('category')
    df['side_placement'] = df['side_placement'].astype('category')
    df['geometry'] = df['geometry'].astype('category')


    # df.drop(columns=["store_id"])
    df = df.drop(columns=['grunnkrets_id', 'plaace_hierarchy_id'])

    return df

label_name = 'revenue'
X = train.drop(columns=[label_name])
y = train[label_name]

# plt.plot([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245], [0.8790265999999999, 0.8681742999999997, 0.8802965999999998, 0.8697140000000001, 0.8717276500000001, 0.8751413500000002, 0.8891620000000001, 0.8748119500000001, 0.8654567, 0.8736065500000001, 0.8723676499999999, 0.87482995, 0.8809048, 0.87815835, 0.8667733000000002, 0.8603021000000002, 0.8798923999999999, 0.8632711500000001, 0.8760869000000001, 0.8621870500000002, 0.8719380500000001, 0.85989575, 0.8796713500000001, 0.8754390999999998, 0.88684925, 0.8672804, 0.8688032000000001, 0.8853343, 0.8698224999999999, 0.8784904000000001, 0.8646378, 0.8720430499999999, 0.8628437999999999, 0.8781760999999999, 0.8702502000000001, 0.8786337499999999, 0.8805001000000001, 0.8790671000000001, 0.8689620000000001, 0.8731875500000001, 0.8830690500000001, 0.8768268499999999, 0.8694514499999999, 0.8752018500000001, 0.8775935500000001, 0.8762643000000001, 0.8742856999999999, 0.87671825, 0.8870310000000001])
# plt.axis([5, 245, 0.8, 0.9])
# plt.show()

plot_data(generate_features(train))


last_rmsle = []
clusters = 110
rounds = 0

while rounds < 500:
    print("Round:", rounds, "with", clusters, "clusters.")
    progress = dict()
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.8)

    # X_train, X_val = generate_features(X_train), generate_features(X_val)
    # KMEANS 189 BEST?
    X_train, X_val = generate_features(X_train), generate_features(X_val)

    # Clear buffers
    folder = os.path.join(os.getcwd(), 'modeling')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
            print(f'Deleted file: {file_path}')

    train_buffer_path = 'modeling/train.buffer'
    test_buffer_path = 'modeling/test.buffer'

    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
    dtrain.save_binary(train_buffer_path)
    print(f'--> {train_buffer_path} created and saved.')

    dvalid = xgb.DMatrix(data=X_val, label=y_val, enable_categorical=True)
    dvalid.save_binary(test_buffer_path)
    print(f'--> {test_buffer_path} created and saved.')

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
    from xgboost import XGBClassifier


    print("Attempting to initialize parameters for training...")
    # params = {
    #     'max_depth': 9,
    #     'eta': 0.05,
    #     'min_child_weight': 7,
    #     'disable_default_eval_metric': True,
    # }
    # params = {'colsample_bytree': 0.5478255177656529, 'learning_rate': 0.02853786979050646, 'max_depth': 7, 'min_child_weight': 1, 'n_estimators': 195, 'subsample': 0.9733176496063143}
    params = {'colsample_bytree': 0.7717138210314867, 'learning_rate': 0.047506668950627134, 'max_depth': 8, 'min_child_weight': 3, 'n_estimators': 223, 'subsample': 0.9929036803032936}
    params['disable_default_eval_metric'] = True
    # print("--> parameters for training initialized.")

    num_round = 999
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


    print("Attempting to start training...")

    bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_round,
            obj=squared_log,
            custom_metric=rmsle,
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval=5,
            evals_result=progress)
    if progress['valid']['RMSLE'][-1] < 0.8:
        last_rmsle.append([rounds, progress['valid']['RMSLE'][-1]])
        break
    print("--> model trained.")
    rounds += 1
print(last_rmsle)


print("Attempting to save model...")
bst.save_model(model_to_load)
print("--> model saved.")

from utils import xgb_cross_validation

params = {
    'max_depth': range(9, 12),
    'eta': [0.05, 0.1, 0.2, 0.3],
    'min_child_weight': range(5, 8),
    # 'disable_default_eval_metric': True,
}

X_test = generate_features(test)
dtest = xgb.DMatrix(data=X_test, enable_categorical=True)

print("\nAttempting to start prediction...")
y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
print("--> Prediction finished.")

print("\nAttempting to save prediction...")
submission['predicted'] = np.array(y_pred)
submission.to_csv('submissions/submission.csv', index=False)
print("--> prediction saved with features as name in submission folder.")

