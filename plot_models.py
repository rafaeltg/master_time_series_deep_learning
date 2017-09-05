#!/usr/bin/env python3.5

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydl.datasets import acf
from pydl.datasets import mackey_glass, create_dataset
from pydl.model_selection import r2_score, rmse, mae
from pydl.models import MLP, RNN, StackedAutoencoder, Autoencoder, DenoisingAutoencoder
from pydl.datasets import test_stationarity

models = ['mlp']


def run_model(m, x_train, y_train, x_test, y_test, data_name):
    print('Optimizing...')

    print('Training')
    m.fit(x_train=x_train, y_train=y_train)

    print('Predict train')
    y_train_pred = m.predict(x_train)

    # Train
    calc_scores(y_true=y_train,
                y_pred=y_train_pred,
                to_file='data/%s_%s_scores_train.csv' % (m.name, data_name))

    plot_preds(y_true=y_train,
               y_pred=y_train_pred,
               title='%s - %s train' % (m.name, data_name),
               to_file='data/%s_%s_train' % (m.name, data_name))

    # Test
    print('Predict test')
    y_test_pred = m.predict(x_test)

    calc_scores(y_true=y_test,
                y_pred=y_test_pred,
                to_file='data/%s_%s_scores_test.csv' % (m.name, data_name))

    plot_preds(y_true=y_test,
               y_pred=y_test_pred,
               title='%s - %s test' % (m.name, data_name),
               to_file='data/%s_%s_test' % (m.name, data_name))

    print('Done')


def calc_scores(y_true, y_pred, to_file=''):

    scores = pd.Series(
        data=[
            r2_score(y_true, y_pred),
            rmse(y_true, y_pred),
            mae(y_true, y_pred),
        ],
        index=[
            'R2',
            'RMSE',
            'MAE',
        ])

    if to_file != '':
        scores.to_csv(to_file)

    return scores.to_dict()


def plot_preds(y_true, y_pred, title='', to_file=''):

    # Actual x Pred
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)

    if to_file != '':
        plt.savefig(filename=to_file + '.png')

    # Errors
    errs = y_true - y_pred

    f, axarr = plt.subplots(2)
    axarr[0].scatter(y_pred, errs)
    axarr[0].set_title('Forecast Residual')
    axarr[0].set_xlabel('Predicted')
    axarr[0].set_ylabel('Residual')
    axarr[0].axhline(y=0, color='black', ls='--')

    axarr[1].set_title('Forecast Residual Autocorrelation')
    acf(pd.Series(np.reshape(errs, len(errs))), nlags=50, plot=True, ax=axarr[1])

    f.subplots_adjust(hspace=0.5)

    if to_file != '':
        f.savefig(filename=to_file + '_errors.png')


def get_models():

    return [
        RNN(name='LSTM',
            layers=[25, 25, 25],
            #dropout=[0., 0.2, 0.2],
            recurrent_dropout=[0.2, 0.2, 0.2],
            stateful=False,
            time_steps=1,
            cell_type='lstm',
            nb_epochs=150,
            batch_size=50),
    ]

    return [
        MLP(name='MLP',
            layers=[50, 50],
            dropout=0.1,
            nb_epochs=100),

        RNN(name='LSTM',
            layers=[50, 50],
            dropout=[0., 0.2],
            recurrent_dropout=[0.2, 0.2],
            stateful=False,
            time_steps=1,
            cell_type='lstm',
            nb_epochs=100,
            batch_size=50),

        StackedAutoencoder(
            name='SAE',
            layers=[Autoencoder(n_hidden=32),
                    Autoencoder(n_hidden=32)],
            dropout=0.1,
            nb_epochs=100),

        StackedAutoencoder(
            name='SDAE',
            layers=[DenoisingAutoencoder(n_hidden=32, corr_param=0.2),
                    DenoisingAutoencoder(n_hidden=16, corr_param=0.1)],
            nb_epochs=100)
    ]


def create_features(ts):

    """
    Features:
        - 10 most correlated lags

    :param ts:
    :return:
    """

    acfs, conf = acf(ts, 100)
    acfs = acfs[1:]

    most_corr = [v > conf for v in acfs]

    corr_lags = np.argsort(acfs[most_corr])[-10:]
    corr_lags = sorted(corr_lags)

    look_back = corr_lags[-1] + 1
    x, y = create_dataset(ts, look_back)

    # use only the most correlated lags
    x = np.array([X[corr_lags] for X in x])
    return x, y


def run_mg():

    """

        Number of samples: 2000
        Train size = 80% (1600 samples)
        Test size = 20% (400 samples)

    """

    n_samples = 2000
    train_size = .8

    # Create time series data
    ts = mackey_glass(sample_len=n_samples)

    print(ts.describe())
    #test_stationarity(ts.as_matrix())

    x, y = create_features(ts)

    # split into train and test sets
    train_idx = int(len(x) * train_size)
    x_train, y_train = x[0:train_idx], y[0:train_idx]
    x_test, y_test = x[train_idx:len(x)], y[train_idx:len(y)]

    for m in get_models():
        run_model(m, x_train, y_train, x_test, y_test, 'Mackey-Glass')


if __name__ == '__main__':
    run_mg()
