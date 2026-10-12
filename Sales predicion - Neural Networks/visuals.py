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

def draw_heatmap(data:pd.DataFrame, **kwargs)->None:
    '''
    Draws heatmap
    
     data: dataframe with correlations per each id

    '''
    df = data.drop('unique_id', axis=1)
    sns.heatmap(df, square=True, vmin = -1, vmax = 1, annot=True,**kwargs)

def show_predictions(true:pd.DataFrame, pred:pd.DataFrame, naive_fc:pd.DataFrame=None) -> None:

    '''
    Plots predictions for each series 

    true: dataframe with columns 'unique_id', 'ds', 'y' - true values per series
    pred: dataframe with prediction for each series
  
    

    '''

    if len(true['unique_id'].unique())>1:

        fig, axs = plt.subplots(22,1, figsize = (10,22*7))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle('Forecast vs Actuals', y = 0.885)
        for id, ax in zip(true['unique_id'].unique(),axs.ravel()):

            ax.set_title(id)
            sns.lineplot(y = true[true['unique_id']==id]['y'], x = true[true['unique_id']==id].index, label = 'Actuals',ax = ax)
            sns.lineplot(y = pred[pred['unique_id']==id].iloc[:,-3].values,x = true[true['unique_id']==id].index, label = 'NBEATS',ax = ax)
            sns.lineplot(y = pred[pred['unique_id']==id].iloc[:,-2].values,x = true[true['unique_id']==id].index, label = 'LSTM',ax = ax)
            sns.lineplot(y = pred[pred['unique_id']==id].iloc[:,-1].values,x = true[true['unique_id']==id].index, label = 'TFT',ax = ax)
            
            if naive_fc is not None:
                sns.lineplot(y = naive_fc[naive_fc['unique_id']==id].iloc[:,-1].values,x = true[true['unique_id']==id].index, label = 'Naive',ax = ax)
            ax.set_ylabel('Item_cnt_day')
            ax.legend(loc = 'upper right')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_ylim(0,np.max(true[true['unique_id']==id]['y'])*1.2)
            
        plt.show()

    else:
        plt.figure(figsize=(8,5))
        sns.lineplot(y = true['y'].values,x = true.index, label = 'Actuals')
        sns.lineplot(y = pred.iloc[:,-3],x = true.index, label = 'MLP')
        sns.lineplot(y = pred.iloc[:,-2],x = true.index, label = 'CNN')
        sns.lineplot(y = pred.iloc[:,-1],x = true.index, label = 'NP')
        sns.lineplot(y = naive_fc.iloc[:,-1].values,x = true.index, label = 'Naive')
        plt.ylim(0,np.max( true['y'].values)*1.2)
        plt.title('Forecast vs Actuals')
        plt.xticks(rotation=90)
        plt.legend(loc='upper right')
        plt.show()

