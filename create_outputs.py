#!/usr/bin/env python3.6

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts import load_npy, acf


"""
    1. Forecasts
        1.1) actual x predicted
        1.2) metrics:
            1.2.1) Train set: CV mean and std of all metrics (mse, rmse, mae, r2)
            1.2.2) Test set: rmse, r2, corr
        1.3) scatter (actual x predicted)
        
    2. Errors
        2.1) box-plot of models
        2.2) White noise (errors acf)
        2.3) scatter (predicted x residual)
    
    3. Input data sets
        3.1) Stationarity test
        3.2) Dataset properties (len, min, max, mean, std...)
"""

titles = {
    'mg': 'Mackey-Glass',
    'lorenz': 'Lorenz',
    'energy': 'ISO New England Energy Demand'
}

idx_transf = {
    'mg': lambda x: x,
    'lorenz': lambda x: x,
    'energy': pd.to_datetime
}


def make_plots(models, ds):

    for d in ds:

        y_test = load_npy('data/%s_test_y.npy' % d)[:, 0]
        idxs = load_npy('data/%s_test_y_index.npy' % d)
        idxs = idx_transf[d](idxs)

        model_errors = []

        for m in models:
            y_test_pred = load_npy('results/predict/%s_%s_preds.npy' % (m, d))[:, 0]

            idxs = idxs
            y_test = y_test
            y_test_pred = y_test_pred

            # Actual x Predicted
            plt.figure(1)
            fig, ax = plt.subplots()
            ax.plot(idxs, y_test, color='red', label='Actual')
            ax.plot(idxs, y_test_pred, color='blue', label='Predicted')
            ax.set_xlim(idxs[0], idxs[-1])
            ax.set_ylim(min(min(y_test), min(y_test_pred))*0.8, max(max(y_test), max(y_test_pred))*1.3, auto=True)
            plt.legend(loc='best')
            plt.title(titles[d])
            fig.autofmt_xdate()
            plt.savefig(fname='results/figs/%s_%s_actual_pred.png' % (m, d))
            plt.clf()

            plt.figure(1)
            plt.scatter(y_test, y_test_pred, s=7)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(titles[d])
            plt.tight_layout()
            plt.savefig(fname='results/figs/%s_%s_actual_pred_scatter.png' % (m, d))
            plt.clf()

            # Residuals
            errs = y_test - y_test_pred

            plt.figure(1)
            fig, ax = plt.subplots()
            ax.scatter(y_test_pred, errs, s=7)
            ax.axhline(y=0, color='black', ls='--')
            ax.set_ylim( (min(errs)*1.1) if min(errs) < 0 else 0, max(errs)*1.1)
            plt.title(titles[d] + ' Forecast Residuals')
            plt.xlabel('Predicted')
            plt.ylabel('Residual')
            plt.savefig(fname='results/figs/%s_%s_residuals_scatter.png' % (m, d))
            plt.clf()

            plt.figure(1)
            plt.title('Forecast Residual Autocorrelation')
            acf(pd.Series(np.reshape(errs, len(errs))), nlags=50, plot=True, ax=plt.gca())
            plt.savefig(fname='results/figs/%s_%s_residuals_acf.png' % (m, d))
            plt.clf()

            model_errors.append(errs)

        # Residuals boxplot
        plt.figure(1)
        plt.boxplot(model_errors, labels=[m.upper() for m in models])
        plt.title(titles[d] + ' Forecast Residuals')
        plt.ylabel('Residual')
        plt.xlabel('Model')
        plt.savefig(fname='results/figs/%s_residuals_boxplot.png' % d)
        plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create output plots and csvs.')
    parser.add_argument('--datasets', dest='datasets', nargs='+', required=True, help='Datasets to be used')
    parser.add_argument('--models', dest='models', nargs='+', required=True, help='Models to create inputs')
    args = parser.parse_args()

    make_plots(args.models, args.datasets)
