import json
import pandas as pd
import numpy as np
import os
import re
import pickle
import catboost
import lightgbm as lgb
import torch
import time
import optuna
from IPython.display import display
from pathlib import Path
from modules.optimization import *
from modules.preprocessing import *
from modules.mlp import *
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler


import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'train.csv'
MODEL_PATH = PROJECT_ROOT / 'artifacts'
CONFIG_PATH = PROJECT_ROOT / 'config.json'

def recall_eval(preds, data):
    y_true = data.get_label()
    y_pred = (preds >= 0.5).astype(int)

    recall = recall_score(y_true, y_pred)
    return "recall", recall, True
    

def train(config:json)->None:
    """ Trains model, optimizes parameters, prints metrics for all models and saves the best model's artifacts"""

    X = pd.read_csv(DATA_PATH)

    train, val = train_test_split(X, test_size=0.25, stratify=X['Response'], random_state=42)
    tr = DataTransformer()
    train = tr.fit_transform(train)
    val = tr.transform(val)

    X_train, X_val, y_train, y_val = train.loc[:,~train.columns.isin(['Response'])], val.loc[:,~val.columns.isin(['Response'])], train['Response'], val['Response']

    metrics={}
    models={}
    features={}

    for model in ['CatBoost','LightGBM', 'MLP']:

        if model=='CatBoost':
            params={
                'loss_function':'Logloss',
                'random_seed':42,
                'verbose': 250,
                'cat_features':tr.cat_idx
                }
            

            if config['method'] is not None:
                print(f'Optimizing {model} with method {config["method"]}')
                time.sleep(5)
                study = optuna.create_study(sampler=TPESampler(), direction='maximize')
                study.optimize(Objective_optuna(model, X_train, X_val, y_train, y_val, config['method']), n_trials=config['trials'], show_progress_bar=True)
                params=study.best_params

                print(f'Training {model}')
                idx = [str(i) for i in range(X_train.shape[1])]

                f = [params[i] for i in idx]

                for i in idx:
                    del params[i]

                samples = [int(re.search(r'\d+',key).group()) for key in params.keys() if 'sample_' in key]
                if len(samples) > 0:
                    for j in samples:
                        f[j] = True
                        del params[f'sample_{j}']

                features[model] = f
                X_selected = X_train.iloc[:, f]

                cat_features_idx = [
                    i for i, col in enumerate(X_selected.columns) 
                    if col in tr.cat_cols
                ]


                if config['method']=='sampling':

                    strategy = params['sampling_strategy']
                    del params['sampling_strategy']

                    pipe = imbPipeline([
                                ('smote', SMOTENC(categorical_features=cat_features_idx,
                                  random_state=42, sampling_strategy=strategy)),
                                  ('cb', CatBoostClassifier(cat_features=tuple(cat_features_idx),**params))
                    ])

                    pipe.fit(X_selected, y_train)
                    pred = pipe.predict(X_val.iloc[:, f])

                    mod = pipe
                    

                if config['method']=='weights':
                    m = CatBoostClassifier(cat_features=tuple(cat_features_idx), **params)
                    m.fit(X_selected, y_train)
                    pred = m.predict(X_val.iloc[:, f])

                    mod = m

                if config['method']=='prob':

                    threshold = params['threshold']
                    del params['threshold']
                    m = CatBoostClassifier(cat_features=tuple(cat_features_idx), **params)
                    m.fit(X_selected, y_train)
                    pred = m.predict_proba(X_val.iloc[:, f])
                    pred = (pred[:,1]>threshold).astype('int')

                    mod = (m, threshold)

            else:

                print(f'Training {model}')
                setup = params.copy()

                m = CatBoostClassifier(**setup)
                m.fit(X_train,
                        y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=10,
                        use_best_model=True
                )

                pred=m.predict(X_val)

                mod = m
                
            metrics[model] = recall_score(y_val,pred)
            models[model] = mod

        if model=='LightGBM':
            params={
                    'objective': 'binary',
                    'seed':42,
                    'verbosity': 1,
                    'categorical_feature':tr.cat_idx
                }
            

            if config['method'] is not None:
                print(f'Optimizing {model} with method {config["method"]}')
                time.sleep(5)
                study = optuna.create_study(sampler=TPESampler(), direction='maximize')
                study.optimize(Objective_optuna(model, X_train, X_val, y_train, y_val, config['method']), n_trials=config['trials'], show_progress_bar=True)
                params=study.best_params

                print(f'Training {model}')
                idx = [str(i) for i in range(X_train.shape[1])]

                f = [params[i] for i in idx]
                for i in idx:
                    del params[i]

                samples = [int(re.search(r'\d+',key).group()) for key in params.keys() if 'sample_' in key]
                if len(samples) > 0:
                    for j in samples:
                        f[j] = True
                        del params[f'sample_{j}']

                features[model] = f
                X_selected = X_train.iloc[:, f]
                cat_features_idx = [
                    i for i, col in enumerate(X_selected.columns) 
                    if col in tr.cat_cols
                ]

                if config['method']=='sampling':

                    strategy = params['sampling_strategy']
                    del params['sampling_strategy']

                    pipe = imbPipeline([
                                ('smote', SMOTENC(categorical_features=cat_features_idx,
                                  random_state=42,sampling_strategy=strategy)),
                                ('lgbm', LGBMClassifier(**params))
                    ])

                    pipe.fit(X_selected, y_train)
                    pred = pipe.predict(X_val.iloc[:, f])

                    mod = pipe
                    

                if config['method']=='weights':
                    m = LGBMClassifier(**params)
                    m.fit(X_selected, y_train)
                    pred = m.predict(X_val.iloc[:, f])

                    mod = m

                if config['method']=='prob':

                    threshold = params['threshold']
                    del params['threshold']
                    m = LGBMClassifier(**params)
                    m.fit(X_selected, y_train)
                    pred = m.predict_proba(X_val.iloc[:, f])
                    pred = (pred[:,1]>threshold).astype('int')

                    mod = (m, threshold)


            else:

                print(f'Training {model}')
                setup = params.copy()

                m = LGBMClassifier(**setup)
                m.fit(X_train,
                        y_train
                )

                pred=m.predict(X_val)

                mod = m
                
            metrics[model] = recall_score(y_val,pred)
            models[model] = mod
        
        if model=='MLP':
            params={}

            if config['method'] is not None:
                print(f'Optimizing {model} with method {config["method"]}')
                time.sleep(5)
                study = optuna.create_study(sampler=TPESampler(), direction='maximize')
                study.optimize(Objective_optuna(model, X_train, X_val, y_train, y_val, config['method']), n_trials=config['trials'], show_progress_bar=True)
                params=study.best_params

                print(f'Training {model}')
                idx = [str(i) for i in range(X_train.shape[1])]

                f = [params[i] for i in idx]

                for i in idx:
                    del params[i]

                samples = [int(re.search(r'\d+',key).group()) for key in params.keys() if 'sample_' in key]
                if len(samples) > 0:
                    for j in samples:
                        f[j] = True
                        del params[f'sample_{j}']

                features[model] = f
                X_selected = X_train.iloc[:, f]
                cat_features_idx = [
                    i for i, col in enumerate(X_selected.columns) 
                    if col in tr.cat_cols
                ]
                if config['method']=='sampling':

                    smote_strategy = params['sampling_strategy']
                    del params['sampling_strategy']

                    pipe = imbPipeline([
                                ('smote', SMOTENC(categorical_features=cat_features_idx,
                                  random_state=42,sampling_strategy = smote_strategy))
                    ])

                    X_tr, y_tr = pipe.fit_resample(X_selected, y_train)
                    X_tr = pd.DataFrame(X_tr, columns=X_selected.columns.tolist())

                    m = MLPBinaryClassifier(
                        input_size=X_tr.shape[1],
                        **params
                    )
                    score = m.train(X_tr, y_tr, X_val.iloc[:, f], y_val)

                    pred = m.predict(X_val.iloc[:, f])

                    mod = (pipe, m, m.preprocessor)
                    

                if config['method']=='weights':
                    m = MLPBinaryClassifier(
                        input_size=X_selected.shape[1],
                        **params
                    )
                    score = m.train(X_selected, y_train, X_val.iloc[:, f], y_val)

                    pred = m.predict(X_val.iloc[:, f])

                    mod = (m, m.preprocessor)

                if config['method']=='prob':

                    m = MLPBinaryClassifier(
                        input_size=X_selected.shape[1],
                        **params
                    )
                    score = m.train(X_selected, y_train, X_val.iloc[:, f], y_val)

                    pred = m.predict(X_val.iloc[:, f])

                    mod = (m, m.preprocessor)

            else:

                print(f'Training {model}')
                setup = params.copy()

                m = MLPBinaryClassifier(input_size=X_train.shape[1], **setup)
                score = m.train(X_train, y_train, X_val, y_val)

                pred=m.predict(X_val)

                mod = (m, m.preprocessor)
                
            metrics[model] = recall_score(y_val,pred)
            models[model] = mod
    
    metrics=pd.DataFrame(metrics, index=['Recall'])
    
    # Saving best model
   
    best = metrics.max().idxmax()
    config_inference={'model_name':best}

    if config['method'] is not None:
        method=config['method']
        config_inference['features']  = features[best]
        config_inference['method']=method
    else:
        method='baseline'
        config_inference['method']=method

    
    
    os.makedirs(os.path.join(MODEL_PATH, method), exist_ok=True)

    # Saving DataTransformer
    data_transformer_path = os.path.join(MODEL_PATH, method, 'data_tr.pkl')
    with open(data_transformer_path, 'wb') as f:
        pickle.dump(tr, f)
    config_inference['data_transformer']=data_transformer_path

    # Saving models and other dependencies
    if best in ['CatBoost', 'LightGBM']:
        
        if len(models[best])==1:
            best_path = os.path.join(MODEL_PATH, method, 'model.pkl')
            config_inference['model_path'] = best_path
            with open(best_path, 'wb') as f:
                pickle.dump(models[best], f)
        else:
            best_path_model = os.path.join(MODEL_PATH, method, 'model.pkl')
            config_inference['model_path'] = best_path_model
            config_inference['threshold'] = models[best][1]
            with open(best_path_model, 'wb') as f:
                pickle.dump(models[best][0], f)           
    else:
        
        best_path_model = os.path.join(MODEL_PATH, method, 'model.pth')
        best_path_preprocessor = os.path.join(MODEL_PATH, method, 'preprocessor.pkl')
        best_path_sampling = os.path.join(MODEL_PATH, method, 'sampling.pkl')


        if len(models[best])==3:

            config_inference['model_path'] = best_path_model
            config_inference['preprocessor_path'] = best_path_preprocessor
            config_inference['sampling_path'] = best_path_sampling

            torch.save(models[best][1], best_path_model)
            with open(best_path_sampling, 'wb') as f:
                pickle.dump(models[best][0], f)
            with open(best_path_preprocessor, 'wb') as g:
                pickle.dump(models[best][2], g)
        else:
            
            config_inference['model_path'] = best_path_model
            config_inference['preprocessor_path'] = best_path_preprocessor
        
            torch.save(models[best][0], best_path_model)
            with open(best_path_preprocessor, 'wb') as f:
                pickle.dump(models[best][1], f)            

    # Saving config for inference
    cf_infer_path = os.path.join(MODEL_PATH, method, f'./config_inference_{method}.json')
    with open(cf_infer_path, 'w') as c:
        json.dump(config_inference, c)

    display(metrics)


if __name__ == "__main__":

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    train(config)

