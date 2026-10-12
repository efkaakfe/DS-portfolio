import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


def multiplot(data: pd.DataFrame) -> None:
    '''
    Plots lineplot, histogram and boxplot of given series
    data: series to plot

    '''
    for col in data.columns[1:-2]:

        fig, ax = plt.subplots(1, 3, figsize=(20, 4))
        fig.suptitle(col)
        fig.subplots_adjust(hspace=0.1)
        sns.lineplot(data, y=col, x=data.index, ax=ax[0])
        sns.scatterplot(data[data['anomaly'] == 1], y=col,
                        x=data[data['anomaly'] == 1].index,
                        color='red', ax=ax[0], label='anomaly')
        sns.histplot(data, x=col, hue='anomaly', ax=ax[1])
        ax[1].set_title('Histogram')
        sns.boxplot(data, y=col, hue='anomaly', ax=ax[2])
        ax[2].set_title('Boxplot')

        plt.show()


def show_time(df: pd.DataFrame) -> dict:
    '''
    Print minutes per hour
    Print count of seconds per each minute per eahc hour

    df: dataframe


    Returns dicts with minutes in each hour, and with # of seconds in each hour
    '''

    time = df.groupby([df.datetime.dt.hour, df.datetime.dt.minute]).count()
    time.index.set_names(['hour', 'minute'], inplace=True)
    time.columns = ['#seconds']
    time = time.reset_index()

    h_min = {}
    for h in time.hour.unique():
        h_min[h] = time.loc[time.hour == h, 'minute'].tolist()
        print(f'Hour/minute:\n{h}: {h_min[h]}')
    h_sec = {}
    for h in time.hour.unique():
        h_sec[h] = time.loc[time.hour == h, "#seconds"].tolist()
        print(
            f'Hour/#seconds:\n{h}: \
            {time.loc[time.hour==h,"#seconds"].tolist()}')

    return h_min, h_sec


def show_missing(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns dataframe with columns representing:
      day, hour, minute and missing seconds in each minute
    df: dataframe
    '''

    time = df.groupby([df.datetime.dt.date, df.datetime.dt.hour,
                       df.datetime.dt.minute]).apply(lambda group:
                                                     check_missing(group))

    time.index.set_names(['day', 'hour', 'minute'], inplace=True)
    time = time.reset_index()
    time = time.rename(columns={0: 'missing_seconds'})
    return time


def check_missing(df: pd.DataFrame) -> list:
    '''
    Indentify missing seconds for given dataframe
    df: dataframe
    '''
    range = np.arange(df.datetime.dt.second.values[0],
                      df.datetime.dt.second.values[-1]+1)
    miss = list(set(range)-set(df.datetime.dt.second))

    return miss


def season_check(df: pd.DataFrame, transform: bool = False) -> None:
    '''
    Check and plot multiple seasonalities

    df: dataframe
    transform: whether to tranform series

    '''

    df_seas = df.copy()
    df_seas['Day'] = df_seas.datetime.dt.date
    df_seas['Hour'] = df_seas.datetime.dt.hour
    df_seas['Minute'] = df_seas.datetime.dt.minute
    df_seas['Second'] = df_seas.datetime.dt.second

    for f in df.columns[1:-2].tolist():

        if transform:
            df_seas[f] = PowerTransformer().fit_transform(
                df_seas[f].values.reshape(-1, 1))

        fig, ax = plt.subplots(3, 1, figsize=(20, 20))
        fig.suptitle(f, y=0.9)
        fig.subplots_adjust(hspace=0.5)
        ax[0].set_title('Daily')
        sns.lineplot(y=df_seas[f], x=df_seas['Hour'],
                     hue=df_seas['anomaly'], ax=ax[0])
        ax[0].legend(loc='upper right')
        ax[1].set_title('Hourly')
        sns.boxplot(y=df_seas[f], x=df_seas['Minute'],
                    hue=df_seas['anomaly'], ax=ax[1])
        ax[1].set_xticks(np.arange(0, 61))
        ax[1].legend(loc='upper right')
        ax[2].set_title('Minute')
        sns.boxplot(y=df_seas[f], x=df_seas['Second'],
                    hue=df_seas['anomaly'], ax=ax[2])
        ax[2].legend(loc='upper right')

    plt.show()


def season(df: pd.DataFrame) -> None:
    '''
    Check and plot multiple seasonalities

    df: dataframe

    '''
    df_seas = df.iloc[:, :-2].copy()
    df_seas['Day'] = df_seas.datetime.dt.date
    df_seas['Hour'] = df_seas.datetime.dt.hour
    df_seas['Minute'] = df_seas.datetime.dt.minute
    df_seas['Second'] = df_seas.datetime.dt.second
    # df_seas = df_seas.drop('datetime',axis = 1)

    df_daily = df_seas.groupby(['Day', 'Hour']).mean().reset_index()
    df_hourly = df_seas.groupby(['Day', 'Minute']).mean().reset_index()
    df_minute = df_seas.groupby(['Day', 'Second']).mean().reset_index()

    for f in df.columns[1:-2].tolist():

        # Daily
        fig, ax = plt.subplots(3, 1, figsize=(30, 10))
        fig.suptitle(f'{f}: Daily', y=0.96)
        fig.subplots_adjust(hspace=0.5)

        sns.lineplot(y=df_daily.loc[df_daily.Day == df_seas.Day.unique()[0], f],
                     x=df_daily.loc[df_daily.Day ==
                                    df_seas.Day.unique()[0], 'Hour'],
                     ax=ax[0])
        ax[0].set_title(df_seas.Day.unique()[0])
        ax[0].legend(loc='upper right')

        ax[1].set_title(df_seas.Day.unique()[1])
        sns.lineplot(y=df_daily.loc[df_daily.Day == df_seas.Day.unique()[1], f],
                     x=df_daily.loc[df_daily.Day ==
                                    df_seas.Day.unique()[1], 'Hour'],
                     ax=ax[1])
        ax[1].set_xticks(np.arange(0, 61))
        ax[1].legend(loc='upper right')
        ax[2].set_title(df_seas.Day.unique()[2])
        sns.lineplot(y=df_daily.loc[df_daily.Day == df_seas.Day.unique()[2], f],
                     x=df_daily.loc[df_daily.Day ==
                                    df_seas.Day.unique()[2], 'Hour'],
                     ax=ax[2])
        ax[2].legend(loc='upper right')

        plt.show()

        # Hourly
        fig, ax = plt.subplots(3, 1, figsize=(30, 10))
        fig.suptitle(f'{f}: Hourly', y=0.96)
        fig.subplots_adjust(hspace=0.5)

        ax[0].set_title(df_seas.Day.unique()[0])
        sns.lineplot(y=df_hourly.loc[df_hourly.Day == df_seas.Day.unique()[0], f],
                     x=df_hourly.loc[df_hourly.Day ==
                                     df_seas.Day.unique()[0], 'Minute'],
                     ax=ax[0])

        ax[0].legend(loc='upper right')

        ax[1].set_title(df_seas.Day.unique()[1])
        sns.lineplot(y=df_hourly.loc[df_hourly.Day == df_seas.Day.unique()[1], f],
                     x=df_hourly.loc[df_hourly.Day ==
                                     df_seas.Day.unique()[1], 'Minute'],
                     ax=ax[1])
        ax[1].set_xticks(np.arange(0, 61))
        ax[1].legend(loc='upper right')

        ax[2].set_title(df_seas.Day.unique()[2])
        sns.lineplot(y=df_hourly.loc[df_hourly.Day == df_seas.Day.unique()[2], f],
                     x=df_hourly.loc[df_hourly.Day ==
                                     df_seas.Day.unique()[2], 'Minute'],
                     ax=ax[2])
        ax[2].legend(loc='upper right')

        plt.show()

        # Minute
        fig, ax = plt.subplots(3, 1, figsize=(30, 10))
        fig.suptitle(f'{f}: Minute', y=0.96)
        fig.subplots_adjust(hspace=0.5)

        ax[0].set_title(df_seas.Day.unique()[0])
        sns.lineplot(y=df_minute.loc[df_minute.Day == df_seas.Day.unique()[0], f],
                     x=df_minute.loc[df_minute.Day ==
                                     df_seas.Day.unique()[0], 'Second'],
                     ax=ax[0])

        ax[0].legend(loc='upper right')

        ax[1].set_title(df_seas.Day.unique()[1])
        sns.lineplot(y=df_minute.loc[df_minute.Day == df_seas.Day.unique()[1], f],
                     x=df_minute.loc[df_minute.Day ==
                                     df_seas.Day.unique()[1], 'Second'],
                     ax=ax[1])
        ax[1].set_xticks(np.arange(0, 61))
        ax[1].legend(loc='upper right')

        ax[2].set_title(df_seas.Day.unique()[2])
        sns.lineplot(y=df_minute.loc[df_minute.Day == df_seas.Day.unique()[2], f],
                     x=df_minute.loc[df_minute.Day ==
                                     df_seas.Day.unique()[2], 'Second'],
                     ax=ax[2])
        ax[2].legend(loc='upper right')

        plt.show()

    plt.show()


def plot_features(df: pd.DataFrame) -> None:
    '''
    Plot multiple features in each line

    df: dataframe

    '''
    plt.figure(figsize=(20, 3))
    sns.lineplot(
        df,
        y=df.columns[1],
        x='datetime'
    )

    plt.show()


def count_missing(df: pd.DataFrame) -> list:
    '''
    Counts missing timestamps per day and hour
    df: dataframe to investigate missing
    Returns list with missing dates

    '''

    miss = {}

    for day in df.datetime.dt.date.unique():
        miss[day] = []
        for hour in df.loc[df.datetime.dt.date == day, :].datetime.dt.hour.unique():

            full = pd.date_range(start=df.loc[(df.datetime.dt.date == day) &
                                              (df.datetime.dt.hour == hour),
                                              'datetime'].values[0],
                                 end=df.loc[(df.datetime.dt.date == day) &
                                            (df.datetime.dt.hour == hour),
                                            'datetime'].values[-1],
                                 freq='s')

            miss[day].extend(full.difference(
                df.loc[(df.datetime.dt.date == day) & (
                    df.datetime.dt.hour == hour), 'datetime']
            ))

    return miss


def binary(data: pd.DataFrame) -> None:
    '''
    Plot countplots for binary variables
    data: dataframe to pick binary features from
    '''

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title(data.select_dtypes(object).columns[0])
    sns.countplot(data, x='anomaly', ax=ax[0], stat='percent')
    # ax[0].set_ylabel(data['anomaly'].value_counts(normalize=True))
    ax[1].set_title(data.select_dtypes(object).columns[1])
    sns.countplot(data, x='changepoint', ax=ax[1], stat='percent')
    # ax[1].set_ylabel(data['changepoint'].value_counts(normalize=True))
    plt.show()


def plot_heatmap(data: pd.DataFrame) -> None:
    '''
    Plot heatmap with Spearman correlations
    data: dataframe
    '''
    c = data.corr(method='spearman')
    c = c[c > 0.3]
    plt.figure(figsize=(10, 10))
    sns.heatmap(c, annot=True, vmin=-1, vmax=1)
    plt.show()


def stationarity(data: pd.DataFrame) -> None:
    '''
    Check stationarity of each time series in the dataframe
    data: dataframe
    '''

    for c in data.select_dtypes(np.float64).columns:
        print(f'{c}: ADF: {adfuller(data[c])[1]}, KPSS: {kpss(data[c])[1]}')


def autocor(data: pd.DataFrame) -> None:
    '''
    Plots ACF and PACF plots for every feature
    data: dataframe to pick featues from
    '''

    for c in data.select_dtypes(np.float64).columns:

        fig, ax = plt.subplots(2, 1, figsize=(35, 10))
        fig.suptitle(c)
        ax[0].set_xticks(np.arange(0, 5*60+1, 5))
        plot_acf(data[c], lags=5*60, ax=ax[0], title='ACF')
        ax[1].set_xticks(np.arange(0, 20*60+1, 5))
        plot_pacf(data[c], lags=20*60, ax=ax[1], title='PACF')

    plt.show()


def plot_comp(dr: Pipeline, n: int) -> None:
    '''
    Plot cumulative variance vs number of components
    dr: fitted dimensionality reduction algoirthm
    n: number of components
    '''
    plt.figure(figsize=(10, 5))
    cum_var = np.cumsum(dr.explained_variance_ratio_)
    sns.lineplot(y=cum_var, x=np.arange(1, n+1))
    plt.yticks(np.arange(cum_var[0], 1.1, 0.05))
    plt.xlabel('Nunber of components')
    plt.ylabel('Cumulative variance')
    plt.show()
