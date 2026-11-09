import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.pipeline import  Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier



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

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OrdinalEncoder()
        self.cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        self.cat_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage','Driving_License','Previously_Insured']
        self.num_cols = None

    def fit(self, X, y=None):
        X_tr = X.copy()
        X_tr = X_tr.drop('id', axis=1)
        X_tr.loc[(X_tr['Vehicle_Age']=="> 2 Years")|(X_tr['Vehicle_Age']=="1-2 Year"),"Vehicle_Age"] = ">1Year"
        self.encoder.fit(X_tr[self.cols])
        self.num_cols = X_tr.columns.difference(self.cat_cols).tolist()
        self.cat_idx = [i for i, col in enumerate(X_tr.columns) if col in self.cat_cols]
        return self
    
    def transform(self, X, y = None):
        X_tr = X.copy()
        X_tr = X_tr.drop('id', axis=1)
        X_tr.loc[(X_tr['Vehicle_Age']=="> 2 Years")|(X_tr['Vehicle_Age']=="1-2 Year"),"Vehicle_Age"] = ">1Year"

        X_tr[self.cols] = self.encoder.transform(X_tr[self.cols]).astype('int32')

        for col in self.cat_cols:
            X_tr[col] = X_tr[col].fillna(0).astype('int32')
        return X_tr

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
