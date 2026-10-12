import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna

from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sktime.forecasting.fbprophet import Prophet
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import RMSE
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')


class Objective_cl:
    def __init__(self, df):
        self.df = df

    def __call__(self, trial):
        '''
        Trial function for KMeans optimization
        trial: optuna trial
        Returns clustering score
        '''

        params = {
            'n_clusters': trial.suggest_int('n_clusters', 2, 10),
            'random_state': trial.suggest_int('random_state', 42, 42),
        }

        m = KMeans(**params)
        p = m.fit_predict(self.df)

        try:
            score = silhouette_score(self.df, p)
        except ValueError:
            score = -1

        return score


def opt_cl(n: int, df: pd.DataFrame) -> dict:
    '''
    Optimize clustering for the best clustering and train optimal model
    n: number of trials
    df: dataframe to cluster
    a: type of algorithm
    Returns best score, cluster labels and fitted model
    '''
    sampler = TPESampler(seed=42)

    study1 = optuna.create_study(sampler=sampler, direction='maximize')
    study1.optimize(Objective_cl(df), n_trials=n, n_jobs=-1)

    m = KMeans(**study1.best_params)
    p = m.fit_predict(df)

    results = {
        'score': silhouette_score(df, p),
        'labels': p,
        'model': m
    }

    return results


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cat=False):
        self.cat = cat
        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        if 'units' in X.columns:
            X_transformed = X.loc[:, ['id', 'date', 'units', 'lag28', 'wday', 'month',
                                      'year', 'events', 'snap', 'sell_price', 'missing_price']]
            X_transformed[['units', 'lag28', 'wday', 'month',
                                      'year', 'events', 'snap', 'missing_price']] = X_transformed[['units', 'lag28', 'wday', 'month',
                                      'year', 'events', 'snap', 'missing_price']].astype('int16')
            X_transformed[['sell_price']] = X_transformed[['sell_price']].astype('float16')
        else:
            X_transformed = X.loc[:, ['id', 'date', 'lag28', 'wday', 'month',
                                      'year', 'events', 'snap', 'sell_price', 'missing_price']]
            X_transformed[['lag28', 'wday', 'month',
                                      'year', 'events', 'snap', 'missing_price']] = X_transformed[['lag28', 'wday', 'month',
                                      'year', 'events', 'snap', 'missing_price']].astype('int16')
            X_transformed[['sell_price']] = X_transformed[['sell_price']].astype('float16')
        
        if self.cat:
            X_transformed[['wday','month','snap','missing_price', 'id']] = X_transformed[['wday','month','snap','missing_price', 'id']].astype('category')

        return X_transformed


class DataframeBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        if X.shape[1] == 11:
            X_transformed = pd.DataFrame(X)
            X_transformed.columns=['lag28', 'wday', 'month', 'year', 'events', 'sell_price', 'id', 'date', 'units', 'snap', 'missing_price']

            X_transformed[['lag28','wday', 'month', 'year', 'events','sell_price']] = X_transformed[['lag28','wday', 'month', 'year', 'events','sell_price']].astype('float16')
            X_transformed[['units','snap', 'missing_price']] = X_transformed[['units','snap', 'missing_price']].astype('int16')
        else:
            X_transformed = pd.DataFrame(X)
            X_transformed.columns=['lag28', 'wday', 'month', 'year', 'events', 'sell_price', 'id', 'date', 'snap', 'missing_price']

            X_transformed[['lag28','wday', 'month', 'year', 'events','sell_price']] = X_transformed[['lag28','wday', 'month', 'year', 'events','sell_price']].astype('float16')
            X_transformed[['snap', 'missing_price']] = X_transformed[['snap', 'missing_price']].astype('int16')

        return X_transformed
    

def precompute_rmsse_and_weights(train: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates denaminator of RMSSE and weights of each series
    train: train dataframe

    Returns weights per id and denominator per id
    '''

    # RMSSE denominator per id
    denom = train.groupby('id')['units'].apply(
        lambda x: np.mean(np.diff(x) ** 2))
    denom = denom.replace(0, np.nan)  # avoid division by zero

    # Weights per id 
    last_28 = train.groupby('id').tail(28)
    last_28['dollars'] = last_28['units'] * last_28['sell_price']
    weights = last_28.groupby('id')['dollars'].sum()

    return denom, weights


def wrmsse(test: pd.DataFrame, pred: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate WRMSSE for each model's prediction
    test, train: dataframes with test and train data
    pred: dataframe with predictions (may contain more than one prediction column)

    Returns dataframe with WRMSSE for each model
    '''

    denom, weights = precompute_rmsse_and_weights(train)

    # Dla każdej kolumny predykcji
    metrics = {}

    if type(pred)==pd.DataFrame:


        for col in pred.columns[2:]:
            merged = test[['id', 'date', 'units']].merge(
                pred[['id', 'date', col]], on=['id', 'date'], how='left')
            merged['diff2'] = (merged['units'] - merged[col]) ** 2
            mse = merged.groupby('id')['diff2'].mean()
            rmsse = np.sqrt(mse / denom)
            wrmsse = np.nansum(rmsse * weights)/ np.nansum(weights)
            metrics[col] = wrmsse

    else:
        merged = test[['id','units']]
        merged['LGBM'] = pred
        merged['diff2'] = (merged['units'] - merged['LGBM']) ** 2
        mse = merged.groupby('id')['diff2'].mean()
        rmsse = np.sqrt(mse / denom)
        wrmsse = np.nansum(rmsse * weights)/ np.nansum(weights)
        metrics['LGBM'] = wrmsse

    return pd.DataFrame(metrics, index=['WRMSSE'])


def create_eval(val: pd.DataFrame, calendar: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    '''
    Create evaluation dataset
    val, calendar, snap: dataframes with validation dataset, calendar and snap data

    Returns evaluation dataset
    '''

    eval = val.loc[:, ['id', 'date', 'state_id']]
    eval['lag28'] = val['units']
    eval['date'] = eval['date']+datetime.timedelta(days=28)
    eval = eval.merge(
        calendar.loc[:, ['date', 'wday', 'month', 'year', 'events']], how='left', on='date')
    eval = eval.merge(snap.loc[:, ['date', 'state_id', 'snap']], how='left', on=[
                      'date', 'state_id'])
    eval = eval.merge(val.iloc[-30490:,:][['id','sell_price','missing_price']]
                      [['id', 'sell_price', 'missing_price']], how='left', on=['id'])

    eval['units'] = -1
    eval['units'] = eval['units'].astype('int16')
    eval['id'] = eval['id'].apply(
        lambda x: x.replace('validation', 'evaluation'))

    return eval


def make_forecast(val: pd.DataFrame, eval: pd.DataFrame, name: str) -> pd.DataFrame:
    '''
    Creates submission csv file and returns dataframe with predictions for the validation and evaluaiton datasets
    val: predictions of the validation set
    eval: predictions of the evaluation set
    name: name of the file
    '''

    df = pd.concat([val, eval])

    submission = pd.DataFrame(df.iloc[:, -1].values.T.reshape(len(df.id.unique()), 28),
                              index=df.id.unique(),
                              columns=['F'+str(i) for i in range(1, 29)]
                              )

    submission = submission.reset_index()
    submission = submission.rename(columns={'index': 'id'})

    submission.to_csv(name, index=False)

    return submission


class Objective_prop:
    def __init__(self, train, splitter, type, pipe):
        self.train = train
        self.splitter = splitter
        self.type = type
        self.pipe = pipe

        if self.type == 'u':
            self.train2 = self.train.set_index(['id', 'date'])[['units']]
        else:
            self.X = self.train.loc[:,~self.train.columns.isin(['units'])]
            self.y = self.train[['id', 'date','units']]
            self.y = self.y.set_index(['id', 'date'])

    def cv(self, model):

        mapes = 0

        if self.type == 'u':

            for i, j in self.splitter:

                i_index = self.train2.index[i]
                m = model.fit(self.train2.loc[i_index])

                pred = model.predict(fh=range(1, 29)).reset_index()
                pred = pred.rename(columns={'units': 'Prophet'})
                pred.iloc[:, -1] = np.round(pred.iloc[:, -1]).astype('int32')
                mapes = mapes + \
                    wrmsse(self.train.iloc[j], pred,
                           self.train.iloc[i]).values[0]

            mapes = mapes/len(self.splitter)
        else:

            for i, j in self.splitter:

                X_tr = self.pipe.fit_transform(
                    self.X.iloc[i]).set_index(['id', 'date'])
                X_t = self.pipe.transform(
                    self.X.iloc[j]).set_index(['id', 'date'])

                # i_index = X_tr.index[i]
                # j_index = X_t.index[j]
                print('Fitting fold')
                m = model.fit(self.y.iloc[i], X=X_tr)
                print('Predicting fold')
                pred = model.predict(fh=range(1, 29), X=X_t).reset_index()
                pred = pred.rename(columns={'units': 'Prophet'})
                pred.iloc[:, -1] = np.round(pred.iloc[:, -1]).astype('int32')
                mapes = mapes + \
                    wrmsse(self.train.iloc[j], pred,
                           self.train.iloc[i]).values[0]

            mapes = mapes/len(self.splitter)

        return mapes

    def __call__(self, trial):

        params = {
            
            'daily_seasonality': trial.suggest_categorical('daily_seasonality', [False]),
            'weekly_seasonality': trial.suggest_int('weekly_seasonality', 2, 7),
            'yearly_seasonality': trial.suggest_int('yearly_seasonality', 7, 15),
            'seasonality_mode': trial.suggest_categorical("seasonality_mode", ['additive', 'multiplicative']),
            'changepoint_prior_scale': trial.suggest_float("changepoint_prior_scale", 0.001, 0.5),
            'seasonality_prior_scale': trial.suggest_float("seasonality_prior_scale", 0.01, 10),
            'holidays_prior_scale': trial.suggest_float("holidays_prior_scale", 0.01, 10),
            'changepoint_range': trial.suggest_float("changepoint_range", 0.8, 0.95),
            'add_country_holidays': trial.suggest_categorical('add_country_holidays', [{'country_name': 'US'}]),
            'freq': trial.suggest_categorical('freq', ['D'])
            
        }

        prophet = Prophet(fit_kwargs={'iter':400},
            **params
        )

        scores_mape = self.cv(prophet)

        return scores_mape


def prop_opt(train: pd.DataFrame, test: pd.DataFrame, n: int, splitter: list, type: str, pipe: Pipeline) -> dict:
    '''
    Optimization of the Prophet model and predicting the test set
    train: train dataframe
    test: test dataframe
    n: number of trials
    type: type of series: 'u' - univariate, 'm' - multivariate
    pipe: Pipeline for preprocessing
    train_tr, test_tr: pre-transformed train and test sets for final prediciton

    Returns dict with WRMSSE, predictions, fitted model and best parameters
    '''

    s = TPESampler(seed=42)

    results = {}

    if type == 'u':
        # Univariate

        print(f'Optimizing')
        p = Objective_prop(train, splitter, type, pipe)

        study = optuna.create_study(sampler=s, direction='minimize')
        study.optimize(p, n_trials=n)

        prop = Prophet(fit_kwargs={'iter': 400},
                       **study.best_params)
        print(f'Predicting')
        prop.fit(train.set_index(['id', 'date'])[['units']])
        pred = prop.predict(fh=range(1, 29)).reset_index()
        pred = pred.rename(columns={'units': 'Prophet'})
        pred.iloc[:, -1] = np.round(pred.iloc[:, -1]).astype('int32')

    else:
        # Multivariate

        print(f'Optimizing')
        p = Objective_prop(train, splitter, type, pipe)

        study = optuna.create_study(sampler=s, direction='minimize')
        study.optimize(p, n_trials=n)
        print('Best params:', study.best_params)
        prop = Prophet(fit_kwargs={'iter': 400},
                       **study.best_params)
        print(f'Predicting')

        y_tr = train[['id', 'date', 'units']]
        y_tr = y_tr.set_index(['id', 'date'])

        X_tr = pipe.fit_transform(train.loc[:, ~train.columns.isin(['units'])])
        X_tr = X_tr.set_index(['id', 'date'])
        X_t = pipe.transform(test.loc[:, ~test.columns.isin(['units'])])
        X_t = X_t.set_index(['id', 'date'])

        prop.fit(y_tr, X=X_tr)
        pred = prop.predict(
            fh=range(1, 29), X=X_t).reset_index()
        pred = pred.rename(columns={'units': 'Prophet'})
        pred.iloc[:, -1] = np.round(pred.iloc[:, -1]).astype('int32')

    err = wrmsse(test, pred, train)

    print('Completed')
    results['WRMSEE'] = err
    results['pred'] = pred
    results['model'] = prop
    results['params'] = study.best_params

    return results


class Objective_lstm:
    def __init__(self, train, splitter, pipe, features):
        self.train = train
        self.splitter = splitter
        self.pipe = pipe
        self.future = features[0]
        # self.hist = features[1]
        self.statics = features[1]

    def cv(self, model):

        mapes = 0

        for i, j in self.splitter:

            m = model.fit(df=self.pipe.fit_transform(self.train.iloc[i]),
                          static_df=self.statics,
                          id_col="id",
                          time_col="date",
                          target_col="units")

            pred = model.predict(
                futr_df=self.pipe.transform(self.train.iloc[j]), static_df=self.statics)
            # pred = pred.rename(columns={'units': 'Prophet'})
            pred.iloc[:, -1] = np.round(pred.iloc[:, -1]).astype('int32')
            mapes = mapes + \
                wrmsse(self.train.iloc[j], pred,
                       self.train.iloc[i]).values[0]

        mapes = mapes/len(self.splitter)

        return mapes

    def __call__(self, trial):

        config_lstm = {
            'h': trial.suggest_int('h', 28, 28),
            'input_size': trial.suggest_int('input_size', 28, 28*4),
            'futr_exog_list': trial.suggest_categorical('futr_exog_list', [self.future]),
            'stat_exog_list': trial.suggest_categorical('stat_exog_list', [self.statics.columns.tolist()[1:]]),
            'encoder_n_layers': trial.suggest_categorical('encoder_n_layers', [2, 4, 8]),
            'encoder_hidden_size': trial.suggest_categorical('encoder_hidden_size', [32, 64, 128]),
            'encoder_dropout': trial.suggest_float('encoder_dropout', 0.0, 0.2),
            'decoder_hidden_size': trial.suggest_categorical('decoder_hidden_size', [32, 64, 128]),
            'decoder_layers': trial.suggest_categorical('decoder_layers', [2, 4, 8]),
            'max_steps': trial.suggest_int('max_steps', 10, 50, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),
            'batch_size': trial.suggest_int('batch_size', 10, 50, 10),
            'windows_batch_size': trial.suggest_categorical('windows_batch_size', [16, 32, 64]),
            'val_check_steps': trial.suggest_categorical('val_check_steps', [10]),
            'scaler_type': trial.suggest_categorical('scaler_type', ['minmax', 'standard', 'robust']),
            'random_seed': trial.suggest_int('random_seed', 10, 100),
            'enable_progress_bar': trial.suggest_categorical('enable_progress_bar', [True]),
            'enable_checkpointing': trial.suggest_categorical('enable_checkpointing', [False])

        }

        net2 = NeuralForecast(models=[
            LSTM(**config_lstm, loss=RMSE())],
            freq='D')

        scores_mape = self.cv(net2)

        return scores_mape


def lstm_opt(train: pd.DataFrame, test: pd.DataFrame, n: int, splitter: list, pipe: Pipeline, features: list) -> dict:
    '''
    Optimization of the Prophet model and predicting the test set
    train: train dataframe
    test: test dataframe
    n: number of trials
    features: list with future, historical feature's names and static dataframe
    pipe: Pipeline for preprocessing
    train_tr, test_tr: pre-transformed train and test sets for final prediciton

    Returns dict with WRMSSE, predictions, fitted model and best parameters
    '''

    s = TPESampler(seed=42)

    results = {}

    print(f'Optimizing')
    p = Objective_lstm(train, splitter, pipe, features)

    study = optuna.create_study(sampler=s, direction='minimize')
    study.optimize(p, n_trials=n)
    print('Best params: ', study.best_params)

    print(f'Predicting')

    net2 = NeuralForecast(models=[
        LSTM(**study.best_params, loss=RMSE())],
        freq='D')

    net2.fit(df=pipe.fit_transform(train),
             static_df=features[1],
             id_col="id",
             time_col="date",
             target_col="units")
    pred = net2.predict(futr_df=pipe.transform(test), static_df=features[1])
    # pred = pred.rename(columns={'units': 'Prophet'})
    pred.iloc[:, -1] = np.round(pred.iloc[:, -1]).astype('int32')

    err = wrmsse(test, pred, train)

    print('Completed')
    results['WRMSEE'] = err
    results['pred'] = pred
    results['model'] = net2
    results['params'] = study.best_params

    return results


class Objective_lgb:
    def __init__(self, train, test, xtrain, ytrain, splitter):
        self. train = train
        self.test = test
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.splitter = splitter
        self.coldim = self.xtrain.shape[1]

    def cv(self, model):

        mapes = 0

        for i, j in self.splitter:

            model.fit(self.xtrain.iloc[i], self.ytrain.iloc[i])

            pred = np.round(model.predict(self.xtrain.iloc[j]))
            # pred = pred.rename(columns={'units': 'Prophet'})
            pred = pred.astype('int32')
            mapes = mapes + \
                wrmsse(self.train.iloc[j], pred,
                       self.train.iloc[i]).values[0]

        mapes = mapes/len(self.splitter)

        return mapes

    def __call__(self, trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 7, 20),
            "min_split_gain": trial.suggest_float("min_split_gain", 0., 0.2),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10., step=0.01),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.4),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0., 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0., 10),
            "random_state": trial.suggest_int("random_state", 42, 42),
            "n_jobs": trial.suggest_int("n_jobs", -1, -1),
            "verbose": trial.suggest_int("verbose", -1, -1)
        }

        model = lgb.LGBMRegressor(**params)

        score = self.cv(model)

        return score


def lgb_opt(train: pd.DataFrame, test: pd.DataFrame, xtrain: pd.DataFrame, ytrain: pd.DataFrame, xtest: pd.DataFrame, ytest: pd.DataFrame, n:int , splitter:list) -> tuple:

    '''
    Optimization and prediction for LGBRegressor

    train, test: train, test dataframes

    X_train, X_test: dataframes with train and test regressors
    y_train, y_test: dataframes with train and test tagrets
    n: number of trials
    splitter: cross-validation splitter

    Returns tuple with error, best parameters, predictions, fitted model and residuals
    '''
 
    s = TPESampler(seed=42)

    print('Optimization start')

    obj2 = Objective_lgb(train, test, xtrain, ytrain, splitter)

    study2 = optuna.create_study(sampler=s, direction='minimize')
    study2.optimize(obj2, n_trials=n)
    print('Best params: ', study2.best_params)
    print('Predicting')

    model = lgb.LGBMRegressor(**study2.best_params)
    model.fit(xtrain, ytrain)

    pred = np.round(model.predict(xtest))
    # pred = pred.rename(columns={'units': 'Prophet'})
    pred = pred.astype('int16')

    res_lgb = ytrain['units'].values - np.round(model.predict(xtrain))

    err = wrmsse(test, pred, train)

    p_lgb = pd.DataFrame({
        'LightGBM': pred,
        'id': xtest['id'].values,
        'date': xtest.index
    })

    results = {}
    results['WRMSSE'] = err
    results['pred'] = p_lgb
    results['model'] = model
    results['params'] = study2.best_params
    results['residuals'] = res_lgb

    print('Completed')

    return results


def format(df: pd.DataFrame, ext=False) -> pd.DataFrame:
    '''
    Format dataset for LGBM model
    df: dataframe to format
    ext: if to format extended dataset
    Returns formatted dataframes for regressors and target
    '''
    df2 = df.drop('missing_price', axis=1)
    df2[['id', 'wday', 'month', 'snap']] = df2[[
        'id', 'wday', 'month', 'snap']].astype('category')

    df2[['sell_price']] = df2[['sell_price']].astype('float32')
    df2[['lag28', 'year']] = df2[['lag28', 'year']].astype('int16')

    if ext:

        df2 = df2.drop('events', axis=1)
        df2[['mean28', 'std28']] = df2[['mean28', 'std28']].astype('float32')

        X = df2.loc[:, ['id', 'date', 'lag28', 'wday', 'month',
                                      'year', 'snap', 'sell_price', 'mean28', 'std28']]
        X = X.set_index('date')

    else:
        X = df2.loc[:, ['id', 'date', 'lag28', 'wday', 'month',
                        'year', 'events', 'snap', 'sell_price']].set_index('date')

    if 'units' in df.columns:
        y = df2[['units']]
    else:
        y = None

    return X, y


def add_rolling(d: pd.DataFrame, tr: pd.DataFrame, n: int) -> pd.DataFrame:
    '''
    Add rolling features to existing dataframe - mean, std
    d: dataframe to compute rolling features from
    tr: dataframe to add rolling features to
    n: rolling window
    '''
    tr_mean = d.groupby('id')['units'].apply(
        lambda x: x.shift(1).rolling(n).mean())
    tr_std = d.groupby('id')['units'].apply(
        lambda x: x.shift(1).rolling(n).std())
    tr_mean.name = 'mean'+str(n)
    tr_std.name = 'std'+str(n)

    train2 = pd.merge(tr, tr_mean.reset_index(level=0)[
                      tr_mean.name], how='left', left_index=True, right_index=True)
    train2 = pd.merge(train2, tr_std.reset_index(level=0)[
                      tr_std.name], how='left', left_index=True, right_index=True)

    return train2


def residual_analysis(res: np.array) -> None:

    ''''
    Returns residual plot, ACF plot and histogram
    Returns mean of residuals, DW statistics, normality test results

    res: array with residuals
    '''

    pd.options.display.float_format = '{:.10f}'.format

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.subplots_adjust(hspace=0.7)
    # Residual plot

    sns.lineplot(res, ax=ax[0])
    ax[0].axhline(0, color='red')
    ax[0].set_title('Residual plot')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    # ACF plot

    plot_acf(res, ax=ax[1], title='ACF plot')

    # Histogram

    sns.histplot(res, kde=True, ax=ax[2])
    ax[2].set_title('Histogram')

    plt.show()

    print(f'Mean of residuals: {np.nanmean(res):.5f}')

    print(f'DW statistic: {durbin_watson(res):.5f}')

    print(
        f'Normality Shapiro test p-value: {stats.shapiro(res, nan_policy="omit")[1]:.5f}')


def insert_vals(X_e: pd.DataFrame, h: pd.DataFrame, pos: int, step=30490) -> pd.DataFrame:
    '''
    Insert rolling mean and std for given position in evaluation dataset

    X_e: evaluation dataset with nans in rolling mean/std columns
    h: target history dataframe
    pos: position to insert values
    step: number of unique series

    Return evaluation dataset with imputted wolling features
    '''
    h = h.groupby('id')
    m = h.apply(lambda x: x['units'][-28:].mean())
    st = h.apply(lambda x: x['units'][-28:].std())
    m = m.reset_index()
    st = st.reset_index()
    vals = X_e.iloc[pos*step:pos*step+step, [0]].merge(m, on='id', how='left')
    vals = vals.merge(st, on='id', how='left')
    vals.columns = ['id', 'mean28', 'std28']
    X_e.iloc[pos*step:pos*step+step, -2:] = vals.iloc[:, -2:].values
    X_e.iloc[pos*step:pos*step+step, -2:] = X_e.iloc[pos *
                                                     step:pos*step+step, -2:].astype('float32')

    return X_e


def recursive(ev: pd.DataFrame, m, h, step=30490) -> pd.DataFrame:
    '''
    Recursive prediction bygiven model
    ev: evaluation set
    m: model
    h: history dataframe
    step: number of unique series

    Return dataframe with predictions
    '''
    n = int(len(ev)/step)
    id = ev.iloc[:step, 0].values

    fc = pd.DataFrame()

    for i in range(n):

        p = np.round(m.predict(ev.iloc[i*step:i*step+step, :]))

        pred = pd.DataFrame({
            'id': id,
            'units': p
        })

        # Update prediction
        fc = pd.concat([fc, pred])
        if i < n-1:

            # Update history
            h = pd.concat([h, pred])
            ev = insert_vals(ev, h, i+1)

    return fc


def count_days(x: pd.Series) -> int:
    '''
    Returns length of unique values in given series
    x: series
    '''
    return len(x.unique())


def price_elast(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Returns dataframe with calculated PED per store per year
    df: dataframe to use for calculations with columns representing
        store_id, item_id, year, date, sales and price
    '''

    # Dollar sales per item
    df['dollars'] = df['units']*df['sell_price']

    # By year
    price_el_groups = df.loc[:, ['store_id', 'item_id', 'year', 'date', 'units', 'dollars']].groupby(['store_id', 'year']).agg({
        'date': count_days,
        'units': 'sum',
        'dollars': 'sum'
    })

    price_el_groups['price'] = price_el_groups['dollars'] / \
        price_el_groups['units']
    price_el_groups['avgunits'] = price_el_groups['units'] / \
        price_el_groups['date']
    price_el_groups = price_el_groups.reset_index(
        level=1)[['year', 'price', 'avgunits']]

    price_el_groups[['price_change', 'units_change']] = price_el_groups[['price', 'avgunits']].apply(
        lambda x: x / x.shift(1)-1
    )
    price_el_groups.loc[price_el_groups.year == 2011,
                        ['price_change', 'units_change']] = np.nan

    price_el_groups['PED'] = price_el_groups['units_change'] / \
        price_el_groups['price_change']

    ped = price_el_groups[['year', 'PED']]

    return ped
