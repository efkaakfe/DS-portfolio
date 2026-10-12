import os
import pandas as pd
import numpy as np
import datetime
import math
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import neuralforecast.models as m
import neuralforecast.auto as na
import statsmodels.api as sm
import optuna
import random
import warnings

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_white
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import STL
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
from optuna.samplers import TPESampler
from ray import tune
from ray.tune.search.optuna import OptunaSearch



warnings.filterwarnings('ignore')

def returns(df:pd.DataFrame)->pd.DataFrame:  
    '''
    Replace returned count with zero and subtract return count from previous day's purchase

    df: original dataframe

    Returns modified dataframe
    
    '''

    df2 = df.copy()
    returns = zip(df2[df2['item_cnt_day']<0]['date'].dt.date.values,df2[df2['item_cnt_day']<0]['unique_id'])

    for date, id in returns:

            return_idx = df2[(df2['unique_id']==id)&(df2['date'].dt.date==date)].index[0]
            filter_max = (df2['unique_id']==id)&(df2['date'].dt.month==date.month)&(df2['date'].dt.year==date.year)
            max_idx = df2[filter_max]['item_cnt_day'].idxmax()
            # Lower max value
            df2.loc[max_idx,'item_cnt_day']+= df2.loc[return_idx,'item_cnt_day']
            # Zero in return
            df2.loc[return_idx,'item_cnt_day'] = 0
    return df2


def modif_z_score(x:pd.Series)->pd.Series:
   
   '''
   Calculate modified z-score
   
    x: series to calculate z-score for

   Returns  absolute values of z-scores

   '''
   x_mid = x.median()
   mad = abs(x - x_mid).median()
   mz = 0.6745 * (x - x_mid) / mad
   return abs(mz)


def extreme(df1:pd.DataFrame, thresholds:numpy.array, ids:numpy.array)->pd.DataFrame:    
    '''
    Identify extreme values representing promotions
    
     df1: dataframe to be modified
     thresholds: array with thresholds to remove outliers for each series
     ids: array with series ids

    Returns dataframe with added 'Promo' column
    '''
    df = df1.copy()

    if 'Promo' in df.columns:
        df = df.drop('Promo', axis = 1)
    extreme = pd.DataFrame()
    for id, f in zip(ids, thresholds):
        d = df.loc[df['unique_id']==id,:]
        extreme = pd.concat([extreme,\
                             ((d['item_cnt_day'] > f*d['item_cnt_day'].shift())&\
                              (d['item_cnt_day'] > f*d['item_cnt_day'].shift(-1)))&\
                                (d['Outlier']==1)&\
                                    (~d['unique_id'].isin([352721,352723,354393,353813]))])
    extreme.columns = ['Promo']
    extreme['Promo'] = extreme['Promo'].astype(int)
    df = pd.concat([df, extreme], axis = 1)

    return df



def clean_data(df:pd.DataFrame, rows:dict, holiday:set)->pd.DataFrame:  

    '''
    Imputting missing values and dropping first/second month with majority of missing values
    
     df: dataframe to be modified
     rows: dict with key represeting each unique_id and value representing list of rows to be removed
     holiday: set with dates of russian holidays

    Returns modified and cleaned dataframe
    '''

    df2 = df.copy()

    matching_month = pd.DataFrame()
    matching_month['Year']  = df.date.dt.year
    matching_month['Month']  = df.date.dt.month
    matching_month['month_id'] = df.date_block_num
    matching_month = matching_month.drop_duplicates().dropna()

    matching_ids = pd.DataFrame()
    matching_ids['unique_id'] = df.unique_id
    matching_ids['shop'] = df.shop_id
    matching_ids['item'] = df.item_id
    matching_ids = matching_ids.drop_duplicates().dropna()

    filter = df2.loc[:,~df2.columns.isin(['item_cnt_day'])].isna().any(axis = 1)
    nan_idx = df2[filter].index.tolist()

    # Day of week
    df2.loc[nan_idx,'dayofweek'] = df2.loc[nan_idx,:]['date'].dt.day_of_week
    # Holiday
    df2.loc[nan_idx,'Holiday']= df2.loc[nan_idx,:]['date'].isin(holiday).astype(int)
    # Promo
    df2.loc[nan_idx,'Promo'] = 0
    
    for i in nan_idx:
        # Date_block_num
        df2.loc[i,'date_block_num']= matching_month.loc[(matching_month['Year']==df2.loc[i,'date'].year)&(matching_month['Month']==df2.loc[i,'date'].month),'month_id'].values
        # Shop id
        df2.loc[i,'shop_id'] = matching_ids.loc[matching_ids['unique_id']==df2.loc[i,'unique_id'],'shop'].values
        # Item id
        df2.loc[i,'item_id'] = matching_ids.loc[matching_ids['unique_id']==df2.loc[i,'unique_id'],'item'].values
        # Item_price - previous value
        df2.loc[i,'item_price']= df2.loc[i-1,'item_price']
    

    # Removing first month
    df_1 = pd.DataFrame()
    for id, block in rows.items():
        df_1 = pd.concat([df2.loc[(df2['unique_id']==id)&(~df2['date_block_num'].isin(block)),:],df_1])
    
    df2 = pd.concat([df2.loc[(~df2['unique_id'].isin(df_1.unique_id.unique())),:],df_1])
    df2 = df2.drop('Missing', axis = 1)

    return df2

def input_target(df:pd.DataFrame)->pd.DataFrame: 

    ''' 

    Imputs target values using seasonal decomposition

    df: datafrme to be modified
    
    '''

    for id in df.unique_id.unique():

        
        series = df.loc[df['unique_id']==id,'item_cnt_day']
        imputed_indices = series[series.isnull()].index

        stl = STL(series.interpolate(), period = 7, robust = True)
        res = stl.fit()

        seasonal_component = res.seasonal
        df_deseasonalised = series - seasonal_component
        df_deseasonalised_imputed = df_deseasonalised.interpolate(method="linear")
        df_imputed = df_deseasonalised_imputed + seasonal_component
        # prevent from negative values
        df_imputed = df_imputed.apply(lambda x: max(0, math.ceil(x)))
        series[imputed_indices] = df_imputed[imputed_indices]
        df.loc[df['unique_id']==id,'item_cnt_day'] = series
    
    return df

def cyclic_features(data:pd.DataFrame, col:str, val:pd.Series, max:int)->pd.DataFrame:

    '''
    Generates and adds cyclic features to the existing dataframe

    data: dataframe to be modified
    col: name of column to generate features from
    val: values to convert to sin/cos 
    max: period of the cyclic feature


    Returns dataframe with added cyclic features
    
    '''

    data[col+'_sin'] = np.sin(2*np.pi*val/max)
    data[col+'_cos'] = np.cos(2*np.pi*val/max)
    return data

def season_check(df:pd.DataFrame, id: str, transform:bool = False)->None:

    '''
    Check and plot multiple seasonalities

    df: dataframe
    id: series id
    transform: whether to transform series

    '''
    
    df_seas = df.copy()
    df_seas['Year'] = df_seas.index.year
    df_seas['Month'] = df_seas.index.month
    df_seas['Week'] = df_seas.index.isocalendar().week
    df_seas['Dayofweek'] = df_seas.index.day_of_week
    df_seas['Dayofyear'] = df_seas.index.dayofyear
    df_seas['Day'] = df_seas.index.day
    

    for f in df.columns.tolist():

        if transform:
            df_seas[f] = PowerTransformer().fit_transform(df_seas[f].values.reshape(-1,1))

        fig, ax = plt.subplots(3,1,figsize = (20,20))

        fig.suptitle(id, y=0.9)
        fig.subplots_adjust(hspace=0.5)

        ax[0].set_title('Weekly')
        sns.boxplot(y = df_seas[f], x = df_seas['Year'], hue = df_seas['Dayofweek'], ax = ax[0])
        ax[0].legend(loc = 'upper right')

        ax[1].set_title('Monthly')
        sns.boxplot(y = df_seas[f], x = df_seas['Day'], ax = ax[1])
        ax[1].legend(loc = 'upper right')

        ax[2].set_title('Yearly')
        sns.boxplot(y = df_seas[f], x = df_seas['Year'], hue = df_seas['Month'], ax = ax[2])

        ax[2].legend(loc = 'upper right')

    plt.show()