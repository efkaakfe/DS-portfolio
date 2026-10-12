import os
import pandas as pd
import numpy as np
import datetime
import math
import random
import warnings
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import optuna
import neuralforecast.models as m
import neuralforecast.auto as na
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_white
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import cross_validate
from sktime.split import SlidingWindowSplitter
from copy import deepcopy
from statsforecast import StatsForecast
from statsforecast.models import RandomWalkWithDrift, Naive, HistoricAverage
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAPE, MAE, RMSE
from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import set_random_seed
from utilsforecast.plotting import plot_series
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')


def get_metrics(true: pd.DataFrame, pred: pd.DataFrame, id: bool = None) -> pd.DataFrame :

    '''
    Returns DataFrame with MAPEs for all models

    true: true values
    pred: predictions
    id: if to include multiple series

    '''

    m = pd.DataFrame()

    if id is not None:

        for id in true['unique_id'].unique():

            m = pd.concat([m, pd.DataFrame({
                'MLP': mean_absolute_percentage_error(true[true['unique_id']==id]['y'],pred[pred['unique_id']==id].iloc[:,-3]),
                'RNN': mean_absolute_percentage_error(true[true['unique_id']==id]['y'],pred[pred['unique_id']==id].iloc[:,-2]),
                'Transformer': mean_absolute_percentage_error(true[true['unique_id']==id]['y'],pred[pred['unique_id']==id].iloc[:,-1])
                }, index =[id])])



    m = pd.concat([m,pd.DataFrame({
        'MLP': mean_absolute_percentage_error(true['y'],pred.iloc[:,-3]),
        'RNN': mean_absolute_percentage_error(true['y'],pred.iloc[:,-2]),
        'Transformer': mean_absolute_percentage_error(true['y'],pred.iloc[:,-1])
    }, index = ['Total MAPE'])
    ])

    return m

def optim(n:int, train: pd.DataFrame, test: pd.DataFrame, true: pd.DataFrame, thresholds:numpy.array, statics: pd.DataFrame = None)-> dict :

    '''
    Optimization of NeuralForecast models

    n: number of trials
    train, test, true: dataframes with train, test datasets and true values
    thresholds: array with MAPEs on the test set from baseline models
    statics: dataframe with encoded static features

    Returns dict with keys: 
        scores: array with MAPEs on test set for each model,
        results: dict where each key is the id of the model, value is config dict of the best model
        predictions: dict where each key is the id of the model, value is the best prediction obtained from that model

    
    '''

    thresholds_ = thresholds
    n_models = len(thresholds_)
    best_scores = thresholds_.copy()
    baseline = thresholds_.copy()
    results ={
        0:None,
        1:None,
        2:None
    }
    best_pred = {
        0:None,
        1:None,
        2:None
    }

    for ls in  [MAE(), MAPE(), RMSE()]:

        net2 = NeuralForecast(models = 
                                [
                                    na.AutoNBEATSx(h =  31,  
                                                num_samples=n,
                                                search_alg=tune.search.optuna.OptunaSearch(), 
                                                loss = ls, 
                                                config = config_nbeats),
                                    na.AutoLSTM(h =  31,  
                                                        num_samples=n,
                                                        search_alg=tune.search.optuna.OptunaSearch(), 
                                                        loss = ls, 
                                                        config = config_lstm),
                                    na.AutoTFT(h =  31,  
                                            num_samples=n,
                                            search_alg=tune.search.optuna.OptunaSearch(), 
                                            loss = ls, 
                                            config = config_tft)  

                                                

                              ],  
                              freq = 'D')
        
        net2.fit(df=train, static_df=statics, val_size=31)
        fc_df = net2.predict(futr_df=test)

        fc_df.iloc[:,-n_models:] = fc_df.iloc[:,-n_models:].map(lambda x: max(0,math.ceil(x)))

        scores = fc_df.iloc[:,-n_models:].apply(lambda x: mean_absolute_percentage_error(true['y'],x), axis = 0).values
       
        f = scores < baseline
        idx = [i for i,x in enumerate(f) if x]

        if f.any():
            baseline[f] = scores[f]
            best_scores[f] = scores[f]
            for i in idx:
                results[i] = net2.models[i].results.get_best_result().config
                best_pred[i] = fc_df.iloc[:,-n_models+i].values

    
    f2 = best_scores == thresholds_
    idx2 = [i for i,x in enumerate(f2) if x]

    if f2.any():
        for i in idx2:
            results[i] = 'default settings'
            best_pred[i] = 'default settings'

    final = dict(
        scores = best_scores,
        params = results,
        predictions = best_pred
    )

    return final




def recursive_prediction(test:pd.DataFrame, model_nf: NeuralForecast = None, step:int = 1, present:pd.DataFrame = None, statics:pd.DataFrame = None) -> pd.DataFrame:

    '''
    Recursive prediction on validation set for NN models

    test: validation set
    model_nf: fitted model of class NauralForecast
    step: length of the prediction set in one iteration
    present: training DataFrame for NeuralForecast model
    statics: static exog features dataframe for NeuralForecast model   

    Returns dataframe with predictions
    
    '''

    if model_nf is not None:
    
        n_models = len(model_nf)

        pred_nf = pd.DataFrame()

        horizon = int(len(test)/len(test['unique_id'].unique()))

        step = len(test['unique_id'].unique())
        

        for j in range(n_models):                  

            pred = pd.DataFrame()
            t = test.copy()
            t['y'] = np.nan

            pr = deepcopy(present)

            values = pr.groupby('unique_id').apply(lambda group: group[-35:]).reset_index(drop=True)[['unique_id','y']]

            future = pd.concat([pr,t.groupby('unique_id').apply(lambda group: group[:1]).reset_index(drop=True)])

            

            for i in range(horizon):
                

                                    
                p1 = model_nf[j].predict(df= pr, static_df = statics, futr_df = future)
                p1.iloc[:,-1] = np.ceil(p1.iloc[:,-1])
                
                if i < horizon-1 :

                    # Update rolling mean, std
                    values.loc[len(values)+i:len(values)+i+step,: ] = p1.iloc[:,[0,2]]

                    mean = values.groupby('unique_id').apply(lambda group: group[['y']].rolling(35).mean().iloc[-1,:]).values
                    st = values.groupby('unique_id').apply(lambda group: group[['y']].rolling(35).std().iloc[-1,:]).values
                    
                    idx1 = t.groupby('unique_id').nth(i+1).index
                    idx2 = t.groupby('unique_id').nth(i).index
                    t.loc[idx1,'Rollingmean35'] = mean
                    t.loc[idx1,'Rollingstd35'] = st
                    t.loc[idx2,'y'] = p1.iloc[:,-1].values

                    

                    # Update df

                    pr = pd.concat([pr,t.groupby('unique_id').apply(lambda group: group[i:i+1]).reset_index(drop=True)])
                    
                    future = pd.concat([pr,t.groupby('unique_id').apply(lambda group: group[i+1:i+2]).reset_index(drop=True)])
                    

                    
                pred = pd.concat([pred,p1])

                    

            pred_nf = pd.concat([pred_nf,pred], axis = 1)

        return pred_nf



    

