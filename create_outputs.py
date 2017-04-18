#!/usr/bin/env python3.5

import math
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pydl.datasets import load_npy


"""
    1. Forecasts
        1.1) actual x predicted (with forecast +- 2*err_std)
        1.2) metrics:
            1.2.1) Train set: CV mean and std of all metrics (rmse, r2)
            1.2.2) Test set: rmse, r2, corr
        1.3) scatter (actual x predicted)
        
    2. Errors
        2.1) box-plot of models
        2.2) scatter
        2.2) White noise (errors acf)
    
    3. Input data sets
        3.1) Stationarity test
        3.2) Train and Test set properties (len, min, max, mean, std...)
"""

models = ['sae', 'sdae', 'mlp']  #'lstm',
data_sets = ['sp500'] # , 'mg', 'energy'


def get_forecast_results():
    for d in data_sets:
        actual = load_npy('data/%s_test_y.npy' % d)[:, 0]
        idxs = load_npy('data/%s_test_y_index.npy' % d)
        idxs = pd.date_range(start=idxs[0], end=idxs[-1])

        for m in models:
            preds = load_npy('results/pred/%s_%s_preds.npy' % (m, d))[:, 0]

            fig, ax = plt.subplots()
            ax.plot(idxs[range(0, len(actual))], actual, color='red', label='Actual')
            ax.plot(idxs[range(0, len(preds))], preds, color='blue', label='Predicted')
            plt.legend(loc='best')

            # format the ticks
            ax.set_xlim(idxs[0], idxs[-1])
            fig.autofmt_xdate()

            plt.savefig(filename='results/figs/%s_%s_actual_preds.png' % (m, d))

            plt.figure()
            plt.scatter(actual[:-1], preds)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(filename='results/figs/%s_%s_preds_scatter.png' % (m, d))


def _get_conf_intervals(errs):
    errs_mean = errs.mean()
    errs_std = errs.std()
    z_critical = stats.norm.ppf(q=0.95)  # Get the z-critical value
    margin_of_error = z_critical * (errs_std/math.sqrt(len(errs)))
    return errs_mean - margin_of_error, errs_mean + margin_of_error


if __name__ == '__main__':
    get_forecast_results()
