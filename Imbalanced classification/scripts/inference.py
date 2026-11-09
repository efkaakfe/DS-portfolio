import math
import pandas as pd
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import os
import pickle
import catboost
import lightgbm as lgb
import mlp
import optimization
import train
from IPython.display import display
from optimization import *
from mlp import *
from train import *
from catboost import CatBoostClassifier
from scipy import stats
from pathlib import Path
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

import warnings
warnings.filterwarnings('ignore')


DATA_PATH = './test.csv'
CONFIG_PATH = './artifacts/prob/config_inference_prob.json'
INFER_PATH = './inference'

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def inference(config_path = CONFIG_PATH):

    config = load_config(config_path)
    
    model_name = config.get('model_name')      
    method = config.get('method')              
    threshold = config.get('threshold', 0.5)   

    data_transformer_path = Path(config.get('data_transformer'))
    model_path = Path(config.get('model_path'))
    features = config.get('features', None)    

    print(f"--- Running inference for: {model_name} (method: {method}) ---")

    df = pd.read_csv(DATA_PATH)
    
    if 'Response' in df.columns:
        y_true = df['Response'].values
        X = df.drop(columns=['Response'])
    else:
        X = df.copy()
        y_true=None

    with open(data_transformer_path, 'rb') as f:
        data_transformer = pickle.load(f)

    X_processed = data_transformer.transform(X)

    if features is not None:
        X_processed = X_processed.iloc[:, features]

    if model_name in ['CatBoost', 'LightGBM']:

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            probs = model.predict_proba(X_processed)[:, 1]
            preds = (probs >= threshold).astype(int)

    elif model_name == 'MLP':

        mlp = torch.load(model_path, weights_only=False)
        preds = mlp.predict(X_processed)

    else:
        raise ValueError(f"Model not recognized: {model_name}")

    if y_true is not None:
        score = recall_score(y_true, preds)

        results_df = pd.DataFrame({
            'Model': [model_name],
            'Method': [method],
            'Threshold': [threshold],
            'Recall_Score': [score]
        })

        display(results_df)
    
    infer_path = os.path.join(INFER_PATH, method)

    os.makedirs(infer_path, exist_ok=True)

    pred_output = pd.DataFrame({'true_label': y_true, 'prediction': preds})
    pred_output.to_csv(os.path.join(infer_path,'predictions.csv'), index=False)

if __name__ == '__main__':
    inference()