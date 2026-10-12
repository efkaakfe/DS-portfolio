
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')

best_model_dir = './best'

class Objective:
    '''
    Objective class for MultinomialNB optimization
    '''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        '''
        Trial function for MultinomialNB optimization
        trial: optuna trial
        Returns optimized score
        '''

        params = {
            'max_features': trial.suggest_int('max_features', 8000, 20000),
            'alpha': trial.suggest_float('alpha', 0.0001, 1),
        }

        m = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=params['max_features'])),
            ('model', MultinomialNB(alpha=params['alpha']))
        ])

        score = np.mean(cross_val_score(m, self.X, self.y, scoring='f1', cv=4))

        return score


def optimize_model(n: int, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) -> dict:
    '''
    Optimize for the best hyperparameters and return results with the best parameters
    n: number of trials
    X_train, X_val, y_train, y_val: dataframes representing features and targets for training and validation sets
    Returns dict with best score, best parameters and predictions
    '''
    sampler = TPESampler(seed=42)

    print('Optimizing')
    study1 = optuna.create_study(sampler=sampler, direction='maximize')
    study1.optimize(Objective(X_train, y_train), n_trials=n, n_jobs=-1)

    print('Evaluating')
    m = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    max_features=study1.best_params['max_features'])),
                ('model', MultinomialNB(alpha=study1.best_params['alpha']))
    ])
    m.fit(X_train, y_train)
    pred = m.predict(X_val)

    # Saving the model
    joblib.dump(m, os.path.join(best_model_dir,'model.joblib'))

    results = {
        'score': classification_report(y_val, pred, digits=4),
        'params': study1.best_params,
        'pred': pred
    }

    return results


def predict_test(X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    '''
    Prediction of the test set and evaluation of the metrics
    X_test, y_test: dataframes for train and test sets

    Returns dict with test score and predictions
    '''
    # Loading fitted model
    m = joblib.load(os.path.join(best_model_dir, 'model.joblib'))

    pred = m.predict(X_test)

    results = {
        'score': classification_report(y_test, pred, digits=4),
        'pred': pred
    }

    return results

