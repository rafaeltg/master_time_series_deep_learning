#!/usr/bin/env python3.6

import argparse
from math import ceil
from pydl.models.utils import save_json, load_json
from pyts import load_npy
from create_datasets import create_datasets
from create_models import models_fn


"""
    CMAES parameters
"""
opt_params = {
    'mg': {
        "pop_size": 24,
        "max_iter": 150,
        "verbose": 10,
        "ftarget": 1e-4,
        "tolfun": 1e-4
    },

    'lorenz': {
        "pop_size": 24,
        "max_iter": 150,
        "verbose": 10,
        "ftarget": 5e-5,
        "tolfun": 5e-5
    },

    'energy': {
        "pop_size": 24,
        "max_iter": 150,
        "verbose": 10,
        "ftarget": 1.5e-4,
        "tolfun": 1.5e-4
    },
}


def create_opt_inputs(model, dt):
    inp = {
        "hp_space": load_json('models/%s_%s.json' % (model, dt)),
        "opt": {
            "method": {
                "class": "cmaes",
                "params": opt_params[dt]
            },
            "obj_fn": {
                "cv": {
                    "method": "split",
                    "params": {
                        "test_size": 0.2
                    },
                    "scoring": "mse"
                }
            },
            "max_threads": 24
        },
        'data_set': {
            "data_x": {
                "path": "data/%s_train_x.npy" % dt,
            },
            "data_y": {
                "path": "data/%s_train_y.npy" % dt,
            }
        }
    }
    save_json(inp, 'inputs/%s_%s_optimize.json' % (model, dt))


def create_fit_inputs(model, dt):
    inp = {
        'model': "results/optimize/%s_%s.json" % (model, dt),
        'data_set': {
            'train_x': {
                'path': "data/%s_train_x.npy" % dt,
            },
            'train_y': {
                'path': "data/%s_train_y.npy" % dt,
            }
        }
    }
    save_json(inp, 'inputs/%s_%s_fit.json' % (model, dt))


def create_cv_inputs(model, dt):

    """
    5-fold sliding window CV

    window = 0.8
    horizon = 0.2
    """

    n_folds = 5
    train_y = load_npy('data/%s_train_y.npy' % dt)
    train_len = len(train_y)

    inp = {
        "model": "results/optimize/%s_%s.json" % (model, dt),

        "cv": {
            "method": "time_series",
            "params": calc_cv_params(train_len, n_folds),
            "max_threads": n_folds,
            "scoring": ["r2", "rmse", "mse", "mae"]
        },

        'data_set': {
            "data_x": {
                "path": "data/%s_train_x.npy" % dt,
            },
            "data_y": {
                "path": "data/%s_train_y.npy" % dt,
            }
        }
    }
    save_json(inp, 'inputs/%s_%s_cv.json' % (model, dt))


def calc_cv_params(dt_size, n_folds, window_size=0.8, horizon_size=0.2):

    """
    window = (window_size / (window_size + n_folds * horizon_size)) * dt_size
    horizon = (horizon_size / (window_size + n_folds * horizon_size)) * dt_size
    """

    aux = window_size + n_folds * horizon_size
    horizon = ceil(dt_size * (horizon_size / aux))
    window = dt_size - n_folds*horizon

    return {
        "window": window,
        "horizon": horizon,
        "fixed": False,
        "by": horizon
    }


def create_pred_inputs(model, dt):
    inp = {
        "model": "results/fit/%s_%s.json" % (model, dt),

        'data_set': {
            "data_x": {
                "path": "data/%s_test_x.npy" % dt,
            }
        }
    }
    save_json(inp, 'inputs/%s_%s_predict.json' % (model, dt))


def create_eval_inputs(model, dt):
    inp = {
        "model": "results/fit/%s_%s.json" % (model, dt),

        'data_set': {
            "data_x": {
                "path": "data/%s_test_x.npy" % dt,
            },
            "data_y": {
                "path": "data/%s_test_y.npy" % dt,
            }
        },

        'scoring': ['rmse', 'mse', 'mae', 'r2'],

        'errors_acf': {
            "nlags": 30
        }
    }

    save_json(inp, 'inputs/%s_%s_eval.json' % (model, dt))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create pydl cli inputs.')
    parser.add_argument('--datasets', dest='datasets', nargs='+', required=True, help='Datasets to be used')
    parser.add_argument('--models', dest='models', nargs='+', required=True, help='Models to create inputs')
    args = parser.parse_args()

    create_datasets(args.datasets)

    for m in args.models:
        model_fn = models_fn[m]

        for d in args.datasets:

            # Create model config
            model_fn(d)

            create_opt_inputs(m, d)
            create_cv_inputs(m, d)
            create_pred_inputs(m, d)
            create_eval_inputs(m, d)
            create_fit_inputs(m, d)
