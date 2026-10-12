import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf

warnings.filterwarnings('ignore')


def grid(col: str) -> None:
    '''
    Plots lineplots, histograms and boxplots in a grid per state, item category and department
    col: columns to plot
    '''
    fg1 = sns.FacetGrid(df, row='state_id', col='cat_id',
                        hue='dept_id', height=10, aspect=2)
    fg1.map(sns.lineplot, 'd', col)
    fg1.add_legend()

    fg2 = sns.FacetGrid(df, row='state_id', col='cat_id',
                        hue='dept_id', height=7, aspect=1)
    fg2.map(sns.histplot, col)
    fg2.add_legend()

    fg3 = sns.FacetGrid(df, row='state_id', col='cat_id',
                        hue='dept_id', height=7, aspect=1)
    fg3.map(sns.boxplot, col, orient='v')
    fg3.add_legend()

    plt.show()


def draw_heatmap(data, **kwargs):
    df = data.loc[:, ['units', 'wday', 'month', 'year',
                      'events', 'snap', 'sell_price']]
    sns.heatmap(df.corr(method='spearman'), vmin=-1, vmax=1, annot=True)


def heatmaps(df: pd.DataFrame) -> None:
    '''
    Plots correlations in total, per state, item category and item department
    df: dataframe 
    '''
    f = ['units', 'wday', 'month', 'year',
         'events', 'snap', 'sell_price']

    states = df.state_id.unique()
    cat = df.cat_id.unique()
    dept = df.dept_id.unique()

    # Total
    df1 = df.loc[:, f]
    plt.title('Overal correlations')
    sns.heatmap(df1.corr(method='spearman'), vmin=-1, vmax=1, annot=True)
    plt.show()

    # Per state
    fig, axs = plt.subplots(1, len(states), figsize=(25, 5))
    for s, a in zip(states, axs.ravel()):
        df2 = df.loc[df.state_id == s, f]
        a.set_title(f'State {s}')
        sns.heatmap(df2.corr(method='spearman'),
                    vmin=-1, vmax=1, annot=True, ax=a)
    plt.show()

    # Per category
    fig, axs = plt.subplots(1, len(cat), figsize=(25, 5))
    for c, a in zip(cat, axs.ravel()):
        a.set_title(f'Category {c}')
        df2 = df.loc[df.cat_id == c, f]
        sns.heatmap(df2.corr(method='spearman'),
                    vmin=-1, vmax=1, annot=True, ax=a)
    plt.show()

    # Per dept
    fig, axs = plt.subplots(2, 4, figsize=(35, 10))
    for c, a in zip(dept, axs.ravel()):
        a.set_title(f'Dept {c}')
        df2 = df.loc[df.dept_id == c, f]
        sns.heatmap(df2.corr(method='spearman'),
                    vmin=-1, vmax=1, annot=True, ax=a)
    fig.delaxes(axs[1][3])
    plt.show()

    # Grid

    fg3 = sns.FacetGrid(df.loc[:, ['dept_id', 'cat_id', 'store_id', 'state_id',
                                   'units', 'wday', 'month', 'year',
                                   'events', 'snap', 'sell_price']], row='state_id', col='cat_id', height=5, aspect=1)
    fg3.map_dataframe(draw_heatmap)
    fg3.add_legend()

    plt.show()


def plot_season(s: dict, df: pd.DataFrame) -> None:
    '''
    Plots seasonality per series in each cluster
    s: dict with keys - clusters and values - series ids
    df: dataframe
    '''
    for key, values in s.items():
        for v in values:
            fig, ax = plt.subplots(1, 2, figsize=(30, 3))
            fig.suptitle(f'{key}|{v}')
            sns.lineplot(y=df.loc[df.id == v, 'units'],
                         x=np.arange(1, len(df.loc[df.id == v, 'units'])+1), ax=ax[0])
            ax[0].set_xticks(
                np.arange(1, len(df.loc[df.id == v, 'units'])+1, 7))
            ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
            plot_pacf(df.loc[df.id == v, 'units'], lags=len(
                df.loc[df.id == v, 'units'])/2-1, ax=ax[1])
            plt.show()
