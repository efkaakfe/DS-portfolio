import math
import pandas as pd
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import os
import tqdm
import pickle
import catboost
import lightgbm as lgb
import mlp
import random
from IPython.display import display
from mlp import *
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler


import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')


DATA_PATH = './train.csv'
MODEL_PATH = './artifacts'

class Objective_optuna:
    def __init__(self, model, X_train, X_val, y_train, y_val, method):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.method = method
        self.cv = StratifiedKFold(n_splits=5)
        self.cat_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage','Driving_License','Previously_Insured']
        self.num_cols = self.X_train.columns.difference(self.cat_cols).tolist()
        self.cat_idx = [
                    i for i, col in enumerate(self.X_train.columns) 
                    if col in self.cat_cols
                ]
        self.num_idx = [i for i in range(self.X_train.shape[1]) if i not in self.cat_idx]

    def cross_val(self, m, threshold, X, y):

        splits = self.cv.split(X, y)
        split_cv = []
        for i,j  in splits:
            split_cv.append((i,j))

        scores = 0
        for x_idx, y_idx in tqdm(split_cv, total = len(split_cv), desc="Cross-validation progress"):

              m.fit(X.iloc[x_idx,:],y.iloc[x_idx])
              pred = m.predict_proba(X.iloc[y_idx,:])
              pred = (pred[:,1]>threshold).astype('int')

              scores+=recall_score(y.iloc[y_idx],pred)
        score = scores/len(split_cv)
        return score

    def __call__(self, trial):
            

        features =  [trial.suggest_categorical(str(i), [True, False]) for i in range(self.X_train.shape[1])]
        num_selected = sum(1 for i in self.num_idx if features[i])
        cat_selected = sum(1 for i in self.cat_idx if features[i])
        if (num_selected > 0) and (cat_selected > 0):
            pass
        elif (num_selected == 0) and (cat_selected > 0):
            sample_numerical = random.sample(self.num_idx, 1)[0]
            features[sample_numerical] = trial.suggest_categorical(f'sample_{sample_numerical}',[True])
        elif (num_selected > 0 ) and (cat_selected == 0):
            sample_categorical = random.sample(self.cat_idx, 1)[0]
            features[sample_categorical] = trial.suggest_categorical(f'sample_{sample_categorical}',[True])
        else:
            sample_numerical = random.sample(self.num_idx, 1)[0]
            sample_categorical = random.sample(self.cat_idx, 1)[0]
            features[sample_numerical] = trial.suggest_categorical(f'sample_{sample_numerical}',[True])
            features[sample_categorical] = trial.suggest_categorical(f'sample_{sample_categorical}',[True])
        
        X_selected = self.X_train.iloc[:, features]
        cat_features_idx = [
            i for i, col in enumerate(X_selected.columns) 
            if col in self.cat_cols
        ]
        if self.method=='sampling':

            if self.model=='CatBoost':


                smote_strategy = trial.suggest_float('sampling_strategy', 0.4, 1.0)
                
                cb_params = {
                    'iterations': trial.suggest_int('iterations', 500, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 20.0),
                    'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
                    'loss_function': trial.suggest_categorical('loss_function',['Logloss']),
                    'random_seed': trial.suggest_categorical('random_seed',[42]),
                    'verbose': trial.suggest_categorical('verbose', [False])
                }
                
                pipe = imbPipeline([
                    ('smote', SMOTENC(categorical_features=cat_features_idx, sampling_strategy=smote_strategy, random_state=42)),
                    ('cb', CatBoostClassifier(cat_features=tuple(cat_features_idx),**cb_params))
                ])

                score = cross_val_score(pipe, X_selected, self.y_train, cv=self.cv, scoring='recall',verbose=2)

                return score.mean()

            if self.model=='LightGBM':

                smote_strategy = trial.suggest_float('sampling_strategy', 0.4, 1.0)

                
                lgbm_params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'random_state': trial.suggest_categorical('random_state', [42]),
                    'verbose': trial.suggest_categorical('verbose',[-1])
                }

                pipe = imbPipeline([
                    ('smote', SMOTENC(categorical_features=cat_features_idx, sampling_strategy=smote_strategy, random_state=42)),
                    ('lgbm', LGBMClassifier(categorical_feature=cat_features_idx,**lgbm_params))
                ])

                score = cross_val_score(pipe, X_selected, self.y_train, cv=self.cv, scoring='recall',verbose=2)

                return score.mean()

            if self.model=='MLP':

                smote_strategy = trial.suggest_float('sampling_strategy', 0.4, 1.0)

                mlp_params = {
                    'n_layers': trial.suggest_int('n_layers', 1, 4),
                    'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.2),
                    'epochs': trial.suggest_int('epochs', 5, 50),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'patience': trial.suggest_int('patience', 3, 10)
                }

                pipe = imbPipeline([
                    ('smote', SMOTENC(categorical_features=cat_features_idx, sampling_strategy=smote_strategy, random_state=42))
                ])
                X_tr, y_tr = pipe.fit_resample(X_selected , self.y_train)
                X_tr = pd.DataFrame(X_tr, columns=X_selected.columns.tolist())

                m = MLPBinaryClassifier(
                    input_size=X_tr.shape[1],
                    **mlp_params
                )
                score = m.train(X_tr, y_tr, self.X_val.iloc[:, features], self.y_val)

                return score

        if self.method=='weights':

            if self.model=='CatBoost':

                # Parameters
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 1000),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 0.001, 0.3
                    ),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'scale_pos_weight': trial.suggest_int('scale_pos_weight', 3, 12),
                    'min_data_in_leaf': trial.suggest_int(
                        'min_data_in_leaf', 1, 100
                    ),
                    'l2_leaf_reg': trial.suggest_float(
                        'l2_leaf_reg', 1e-3, 20.0
                    ),
                    'random_strength': trial.suggest_float(
                        'random_strength', 0.0, 2.0
                    ),
                    'loss_function': trial.suggest_categorical(
                        'loss_function', ['Logloss']
                    ),
                    'random_seed': trial.suggest_categorical(
                        'random_seed', [42]
                    ),
                    'verbose': trial.suggest_categorical(
                        'verbose', [False]
                    )
                }

                m = CatBoostClassifier(cat_features=tuple(cat_features_idx),**params)

                score = cross_val_score(m, X_selected, self.y_train, cv=self.cv, scoring='recall',verbose=2)

                return score.mean()

            if self.model=='LightGBM':

                # Parameters
                params = { 
                    'n_estimators': trial.suggest_int(
                        'n_estimators', 200, 1500
                    ),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 0.001, 0.2
                    ),
                    'num_leaves': trial.suggest_int(
                        'num_leaves', 15, 127
                    ),
                    'max_depth': trial.suggest_int(
                        'max_depth', 3, 12
                    ),
                    'scale_pos_weight': trial.suggest_int('scale_pos_weight', 3, 12),
                    'min_child_samples': trial.suggest_int(
                        'min_child_samples', 5, 100
                    ),
                    'reg_lambda': trial.suggest_float(
                        'reg_lambda', 0, 5
                    ),
                    'reg_alpha': trial.suggest_float(
                        'reg_alpha', 0, 5
                    ),
                    'random_state': trial.suggest_categorical(
                       'random_state', [42]
                    ),
                    'verbose': trial.suggest_categorical('verbose',[-1])
                }
                m = LGBMClassifier(categorical_feature=cat_features_idx,**params)
                score = cross_val_score(m, X_selected, self.y_train, cv=self.cv, scoring='recall',verbose=2)

                return score.mean()

            if self.model=='MLP':

                mlp_params = {
                    'n_layers': trial.suggest_int('n_layers', 1, 4),
                    'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.2),
                    'epochs': trial.suggest_int('epochs', 5, 50),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'patience': trial.suggest_int('patience', 3, 10),
                    'positive_class_weight': trial.suggest_int('positive_class_weight', 3, 12)
                }

                m = MLPBinaryClassifier(
                    input_size=self.X_train.iloc[:, features].shape[1],
                    **mlp_params
                )
                score = m.train(X_selected, self.y_train, self.X_val.iloc[:, features], self.y_val)

                return score

        if self.method=='prob':

            if self.model=='CatBoost':

                # Parameters
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 1000),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 0.001, 0.3
                    ),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'scale_pos_weight': trial.suggest_int('scale_pos_weight', 3, 12),
                    'min_data_in_leaf': trial.suggest_int(
                        'min_data_in_leaf', 1, 100
                    ),
                    'l2_leaf_reg': trial.suggest_float(
                        'l2_leaf_reg', 1e-3, 20.0
                    ),
                    'random_strength': trial.suggest_float(
                        'random_strength', 0.0, 2.0
                    ),
                    'loss_function': trial.suggest_categorical(
                        'loss_function', ['Logloss']
                    ),
                    'random_seed': trial.suggest_categorical(
                        'random_seed', [42]
                    ),
                    'verbose': trial.suggest_categorical(
                        'verbose', [False]
                    )
                }

                threshold = trial.suggest_float('threshold', 0.5, 0.85)

                m = CatBoostClassifier(cat_features=tuple(cat_features_idx),**params)

                score = self.cross_val(m, threshold, X_selected, self.y_train)

                return score

            if self.model=='LightGBM':

                # Parameters
                params = { 
                    'n_estimators': trial.suggest_int(
                        'n_estimators', 200, 1500
                    ),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 0.001, 0.2
                    ),
                    'num_leaves': trial.suggest_int(
                        'num_leaves', 15, 127
                    ),
                    'max_depth': trial.suggest_int(
                        'max_depth', 3, 12
                    ),
                    'scale_pos_weight': trial.suggest_int('scale_pos_weight', 3, 12),
                    'min_child_samples': trial.suggest_int(
                        'min_child_samples', 5, 100
                    ),
                    'reg_lambda': trial.suggest_float(
                        'reg_lambda', 0, 5
                    ),
                    'reg_alpha': trial.suggest_float(
                        'reg_alpha', 0, 5
                    ),
                    'random_state': trial.suggest_categorical(
                       'random_state', [42]
                    ),
                    'verbose': trial.suggest_categorical('verbose',[-1])
                }
                m = LGBMClassifier(categorical_feature=cat_features_idx,**params)

                threshold = trial.suggest_float('threshold', 0.5, 0.85)

                score = self.cross_val(m, threshold, X_selected, self.y_train)

                return score

            if self.model=='MLP':

                mlp_params = {
                    'n_layers': trial.suggest_int('n_layers', 1, 4),
                    'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.2),
                    'epochs': trial.suggest_int('epochs', 5, 50),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'patience': trial.suggest_int('patience', 3, 10),
                    'threshold': trial.suggest_float('threshold', 0.5, 0.85)
                }

                m = MLPBinaryClassifier(
                    input_size=self.X_train.iloc[:, features].shape[1],
                    **mlp_params
                )
                score = m.train(X_selected, self.y_train, self.X_val.iloc[:, features], self.y_val)

                return score   
    


    