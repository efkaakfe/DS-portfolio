
import pandas as pd
import numpy as np
import neuralforecast.auto as na
import optuna
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, f1_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sktime.forecasting.fbprophet import Prophet
from copy import deepcopy
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')


def fit_predict_nf(pipe: Pipeline, nf: NeuralForecast, xtrain: pd.DataFrame, ytrain: pd.Series, xtest: pd.DataFrame, ytest: pd.Series) -> tuple:
    '''
    Preprocessing, fitting model on train set and prediction on test set
    pipe: preprocessing Pipeline
    nf: NeuralForecast object
    xtrain: train dataframe
    ytrain: train target series
    xtest: test dataframe
    ytest: test tagret series

    Returns test score and predictions

    '''

    # Dimensionality reduction and creation of NF datasets - train, test
    x_tr = pd.DataFrame(pipe.fit_transform(xtrain), columns=[
                        'PC'+str(i) for i in range(1, 5)], index=xtrain.index)
    tr = pd.concat([x_tr, ytrain], axis=1)
    tr['unique_id'] = 1
    tr['ds'] = np.arange(0,len(tr))
    tr = tr.rename(columns={'anomaly': 'y'})

    x_t = pd.DataFrame(pipe.transform(xtest), columns=[
                       'PC'+str(i) for i in range(1, 5)], index=xtest.index)
    t = pd.concat([x_t, ytest], axis=1)
    t['unique_id'] = 1
    t['ds'] = np.arange(len(tr), len(tr)+len(t))
    t = t.rename(columns={'anomaly': 'y'})

    # Fitting NF model
    nf.fit(df=tr)

    # Predicting NF model
    pred = nf.predict(futr_df=t)

    # If (prob > threshold) -> 1
    pred.iloc[:, -1] = (pred.iloc[:, -1] > 0.5) * 1

    score = f1_score(ytest, pred.iloc[:, -1])

    return score, pred


def optim(n: int, horizon: int, pipe: Pipeline, xtrain: pd.DataFrame, ytrain: pd.Series, xtest: pd.DataFrame, ytest: pd.Series, c1: dict, statics: pd.DataFrame = None) -> tuple:
    '''
    Optimization of the NeuralForecast models

    n: number of trials
    horizon: forecasting horizon
    pipe: preprocessing pipeline
    xtrain: train dataframe
    ytrain: train target series
    xtest: test dataframe
    ytest: test tagret series
    c1: config dict 
    statics: dataframe with static covariates

    Returns f1 test score, predicitons, model config
    '''

    net2 = NeuralForecast(models=[
        na.AutoLSTM(h=horizon,
                    num_samples=n,
                    search_alg=tune.search.optuna.OptunaSearch(),
                    loss=DistributionLoss('Bernoulli'),
                    config=c1)],
        freq=1)

    score, pred = fit_predict_nf(pipe, net2, xtrain, ytrain, xtest, ytest)
    params = net2.models[0].results.get_best_result().config

    return score, pred, params


def crossval(xtrain: pd.DataFrame, ytrain: pd.Series, pipe: Pipeline, splitter: list, type: str) -> float:
    '''
    Cross validation for unsupervised models
    xtrain: train dataframe
    ytrain: train target series
    pipe: Pipeline to train and evaluate
    splitter: CV splitter
    type: model name

    Returns average f1 score from all folds
    '''

    score = 0

    for i, j in splitter:

        pipe.fit(xtrain.iloc[i, :])
        pred = pipe.predict(xtrain.iloc[j, :])

        if type == 'if':
            pred = np.where(pred < 0, 1, 0)

        score = score + f1_score(ytrain.iloc[j], pred)

    score = score/len(splitter)

    return score


class Objective_ml:
    def __init__(self, xtrain, ytrain, cv, pipe, model):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.cv = cv
        self.pipe = deepcopy(pipe)
        self.model = model

    def __call__(self, trial):

        if self.model == 'rf':

            params = {
                "rf__n_estimators": trial.suggest_int("rf__n_estimators", 100, 500),
                "rf__max_depth": trial.suggest_int("rf__max_depth", 3, 100),
                "rf__min_samples_split": trial.suggest_int("rf__min_samples_split", 2, 50),
                "rf__min_samples_leaf": trial.suggest_int("rf__min_samples_leaf", 1, 50),
                "rf__max_features": trial.suggest_int("rf__max_features", 1, 4),
                "rf__random_state": trial.suggest_int("rf__random_state", 42, 42),
                "rf__n_jobs": trial.suggest_int("rf__n_jobs", -1, -1),
                "rf__verbose": trial.suggest_int("rf__verbose", 0, 0)
            }
        if self.model == 'svm':

            params = {
                "svm__C": trial.suggest_float("svm__C", 0.001, 100.0),
                "svm__kernel": trial.suggest_categorical("svm__kernel", ['linear', 'poly', 'rbf', 'sigmoid']),
                "svm__gamma": trial.suggest_float("svm__gamma", 0.0001, 1.0),
                "svm__degree": trial.suggest_int("svm__degree", 2, 5)
            }

        if self.model == 'if':

            params = {
                "if__n_estimators": trial.suggest_int("if__n_estimators", 100, 500),
                "if__max_samples": trial.suggest_float("if__max_samples", 0.2, 1.0),
                "if__contamination": trial.suggest_float("if__contamination", 0.3, 0.4),
                "if__max_features": trial.suggest_int("if__max_features", 1, 4),
                "if__random_state": trial.suggest_int("if__random_state", 42, 42),
                "if__n_jobs": trial.suggest_int("if__n_jobs", -1, -1),
                "if__verbose": trial.suggest_int("if__verbose", 0, 0)
            }

            self.pipe.set_params(**params)
            score = crossval(self.xtrain, self.ytrain,
                             self.pipe, self.cv, self.model)

            return score

        if self.model == 'ae':

            params = {
                "ae__epoch_num": trial.suggest_int("ae__epoch_num", 10, 100),
                "ae__lr": trial.suggest_float("ae__lr", 0.001, 0.1),
                "ae__contamination": trial.suggest_float("ae__contamination", 0.3, 0.4),
                "ae__batch_size": trial.suggest_categorical("ae__batch_size", [16, 32, 64, 128, 256]),
                "ae__dropout_rate": trial.suggest_float("ae__dropout_rate", 0.1, 0.4),
                "ae__hidden_neuron_list": trial.suggest_categorical("ae__hidden_neuron_list", [[64, 32], [128, 32], [256, 32], [128, 64], [256, 64], [512, 64]]),
                "ae__verbose": trial.suggest_int("ae__verbose", 0, 0)
            }

            self.pipe.set_params(**params)
            score = crossval(self.xtrain, self.ytrain, self.pipe, self.cv, self.model)

            return score
        
        # self.pipe.set_params(selector__idx=indices)
        self.pipe.set_params(**params)
        score = np.mean(cross_val_score(self.pipe,
                                        self.xtrain,
                                        self.ytrain,
                                        scoring='f1',
                                        cv=self.cv))

        return score


def train(n: int, threshold: float, pipe: Pipeline, xtrain: pd.DataFrame, ytrain: pd.Series, xtest: pd.DataFrame, true: pd.Series, type: str, cv: list) -> tuple:
    '''
    Optimization and fitting the best Neural Prophet model

    n: number of trials
    threshold: score on validation set obtained from crossvalidation of the baseline model
    pipe: Pipeline to train and evaluate
    xtrain, ytrain, xtest: training, test sets
    true: true values of the test set
    type: model type
    cv: CV splitter

    Returns f1 test score, predictions, best model parameters, best cv score
    '''
    p = deepcopy(pipe)

    sampler = TPESampler(seed=42)
    i = 1
    init = threshold
    print(f'Optimization with baseline {threshold}')
    while init <= threshold:

        study1 = optuna.create_study(sampler=sampler, direction='maximize')
        study1.optimize(Objective_ml(xtrain, ytrain, cv, p, type), n_trials=n)
        init = study1.best_value
        i += 1
        if i in [20, 40, 60, 80]:
            print(f'Iteration {i}')
        if i > 100:
            print('Too many iterations')
            break

    print('Optimization completed\n')
    p.set_params(**study1.best_params)

    if type in ['rf', 'svm']:
        p.fit(xtrain, ytrain)
        pred = p.predict(xtest)
        best_score = f1_score(true, pred)
    if type == 'ae':
        p.fit(xtrain)
        pred = p.predict(xtest)
        best_score = f1_score(true, pred)
    if type == 'if':
        p.fit(xtrain)
        pred = p.predict(xtest)
        pred = np.where(pred < 0, 1, 0)
        best_score = f1_score(true, pred)

    return best_score, pred, p, study1.best_params, study1.best_value


def val_predict(xtrain: pd.DataFrame, c: dict, n: int, horizon: int) -> pd.DataFrame:
    '''
    Prediction of the validation set using LSTM model
    xtrain: DataFrame with preprocessed features
    c: config dict
    n: number of trials

    Return Dataframe with predicted validation set
    '''

    val_predicted = pd.DataFrame()
    print('Initializing the model')

    net2 = NeuralForecast(models=[
        na.AutoLSTM(h=horizon,
                    num_samples=n,
                    search_alg=tune.search.optuna.OptunaSearch(),
                    config=c)],
        freq=1)

    for i in range(4):
        print(f'Predicting feature {i+1}')
        train = xtrain.iloc[:, [i, -2, -1]]
        train = train.rename(columns={'PC'+str(i+1): 'y'})
        net2.fit(df=train)
        pred = net2.predict().iloc[:, -1:]
        pred = pred.rename(columns={'LSTM': 'PC'+str(i+1)})
        val_predicted = pd.concat([val_predicted, pred], axis=1)

    return val_predicted


def val_detect(xtrain: pd.DataFrame, ytrain: pd.Series, val: pd.DataFrame, models: dict, true: pd.Series) -> pd.DataFrame:
    '''
    Outlier detection using given models
    xtrain: train DataFrame with preprocessed features
    ytrain: train target series
    val: validation dataframe
    models: dict with keys - model shortcuts, values - models
    true: true labels

    Returns dataframe with f1 scores for every model
    '''
    scores = pd.DataFrame()

    for id in models.keys():
        print(f'Forecasting for {id}')
        if id in ['rf', 'svm']:

            models[id].fit(xtrain, ytrain)
            pred = models[id].predict(val)
            sc = f1_score(true, pred)
            scores[id] = [sc]

        if id in ['if', 'ae']:
            models[id].fit(xtrain)
            pred = models[id].predict(val)
            if id == 'if':
                pred = np.where(pred < 0, 1, 0)
            sc = f1_score(true, pred)
            scores[id] = [sc]

        if id == 'lstm':
            tr = pd.concat([xtrain, ytrain.reset_index(drop=True)], axis=1)
            tr['unique_id'] = 1
            tr['ds'] = np.arange(0, len(tr))
            tr = tr.rename(columns={'anomaly': 'y'})

            t = val.copy()
            t['unique_id'] = 1
            t['ds'] = np.arange(len(tr), len(tr)+len(t))

            models[id].fit(df=tr)
            pred = models[id].predict(futr_df=t)
            pred.iloc[:, -1] = (pred.iloc[:, -1] > 0.5) * 1
            sc = f1_score(true, pred.iloc[:, -1])
            scores[id] = [sc]

        scores = scores.rename(columns={
            'rf': 'Random Forest',
            'svm': 'SVC',
            'if': 'Isolation Forest',
            'ae': 'Autoencoder',
            'lstm': 'LSTM'
        })
        scores.index = ['F1_score']

    return scores


class Objective_prop:
    def __init__(self, y, splitter, h):
        self.y = y.copy()
        self.splitter = splitter
        self.h = h

    def cv(self, model):

        mapes = 0
        for i, j in self.splitter:

            m = model.fit(self.y.iloc[i])

            pred = model.predict(fh=np.arange(1, self.h+1))
            mapes = mapes + \
                mean_absolute_error(self.y.iloc[j], pred)

        mapes = mapes/len(self.splitter)

        return mapes

    def __call__(self, trial):

        params = {
            'daily_seasonality': trial.suggest_int('daily_seasonality', 2, 7),
            'weekly_seasonality': trial.suggest_int('weekly_seasonality', 2, 7),
            'yearly_seasonality': trial.suggest_int('yearly_seasonality', 5, 12),
            'seasonality_mode': trial.suggest_categorical("seasonality_mode", ['additive', 'multiplicative']),
            'changepoint_prior_scale': trial.suggest_float("changepoint_prior_scale", 0.001, 0.5),
            'seasonality_prior_scale': trial.suggest_float("seasonality_prior_scale", 0.01, 10.0),
            'holidays_prior_scale': trial.suggest_float("holidays_prior_scale", 0.01, 10.0),
            'changepoint_range': trial.suggest_float("changepoint_range", 0.8, 0.95)
        }

        prophet = Prophet(freq='s',
                          add_seasonality={
                              'name': 'custom',
                              'period': trial.suggest_int('period', 80, 110),
                              'fourier_order': trial.suggest_int('fo', 2, 10)
                          },
                          **params
                          )

        scores_mape = self.cv(prophet)

        return scores_mape


def predict_regressors(train: pd.DataFrame, n: int, splitter: list, model: str, horizon: int) -> pd.DataFrame:
    '''
    Tuning model for each regressor and predicting the test set
    train: training set dataframe
    n: number of trials
    splitter: list with train, test indices
    model: model type 
    horizon: length of the forecast

    Returns dataframe with the predicted validation set

    '''
    s = TPESampler(seed=42)
    train_p = train.copy()
    test = pd.DataFrame()

    if model == 'prophet':

        for col in train_p.columns:
            print(f'Optimizing {col}')
            p = Objective_prop(train_p[col], splitter, horizon)

            study = optuna.create_study(sampler=s, direction='minimize')
            study.optimize(p, n_trials=n)

            params = deepcopy(study.best_params)
            fo = params['fo']
            p = params['period']

            del params['fo']
            del params['period']

            prop = Prophet(freq='s',
                           add_seasonality={
                                'name': 'custom',
                                'period': p,
                                'fourier_order': fo
                           },
                           **params)

            print(f'Predicting {col}')

            pred = prop.fit_predict(train_p[col], fh=range(1, horizon+1))

            test[col] = pred.values

    else:
        # Alternative for future use if needed
        pass

    return test
