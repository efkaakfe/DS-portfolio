import math
import pandas as pd
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import os
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.pipeline import FeatureUnion, Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler


import warnings
warnings.filterwarnings('ignore')

DATA_PATH = './train.csv'

def prepare_data_types(dtf:pd.DataFrame) -> pd.DataFrame:
    """Converts data to correct types"""
    dt = dtf.copy()
    dt.Driving_License=dt.Driving_License.astype('object')
    dt.Response=dt.Response.astype('object')
    dt.Previously_Insured=dt.Previously_Insured.astype('object')
    dt.Policy_Sales_Channel=dt.Policy_Sales_Channel.astype('int32')
    dt.Region_Code=dt.Region_Code.astype('int32')
    dt.Vintage=dt.Vintage.astype('int32')
    dt.Age=dt.Age.astype('int32')
    dt.Annual_Premium=dt.Annual_Premium.astype('float32')
    return dt

def plot_corr(dt:pd.DataFrame)->None:
    """ Plots correlation matrix"""
    cat = dt.select_dtypes('object').columns.tolist()
    num = dt.select_dtypes(['int32','float32']).columns.tolist()
    enc = LabelEncoder()
    df = dt[num]
    for c in cat:
        enc_cat = pd.DataFrame(enc.fit_transform(dt[c]).reshape(-1,),columns=[c])
        df = pd.concat([df,enc_cat],axis=1)
    plt.figure(figsize=(8,8))
    plt.title('Correlations between features')
    sns.heatmap(df.corr(method='spearman').round(2),annot=True,vmin=-1,vmax=1)
    plt.show()
    
def baseline()->DummyClassifier:
    """Trains Dummy Classifier and returns metrics on the test set"""
    # Preprocessing
    train = pd.read_csv(DATA_PATH)
    X, val = train_test_split(train, test_size=0.25, stratify=train['Response'])
    X = prepare_data_types(X)
    val = prepare_data_types(val)
    cat_features = X.select_dtypes('object').columns.tolist()
    cat_features.remove('Response')

    prep = ColumnTransformer(
        [('cat', OneHotEncoder(),cat_features)], 
        remainder='passthrough'
    )
    pipe = Pipeline([
        ('prep', prep),
        ('model', DummyClassifier(strategy='stratified',random_state=42))
    ])
    l = LabelEncoder()
    y_train = l.fit_transform(X['Response'])
    y_val = l.transform(val['Response'])
    pipe.fit(X.loc[:,~X.columns.isin(['Response'])], y_train)
    pred = pipe.predict(val.loc[:,~val.columns.isin(['Response'])])

    return recall_score(y_val,pred)