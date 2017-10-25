#!/usr/bin/env python3.5

import argparse
from pydl.models.utils import save_json, load_json
from create_datasets import create_datasets
from create_models import models_fn


"""
    3-fold sliding window CV
"""
cv_params = {
    'sp500': {
        "window": 3238,
        "horizon": 252,
        "fixed": False,
        "by": 252
    },

    'mg': {
        "window": 2311,
        "horizon": 550,
        "fixed": False,
        "by": 550
    },

    'energy': {
        "window": 11711,
        "horizon": 1440,
        "fixed": False,
        "by": 1440
    }
}


def create_opt_inputs(model, dt):
    inp = {
        "hp_space": load_json('models/%s_%s.json' % (model, dt)),
        "opt": {
            "method": {
                "class": "cmaes",
                "params": {
                    "pop_size": 24,
                    "max_iter": 60,
                    "verbose": 10
                }
            },
            "obj_fn": {
                "cv": {
                    "method": "split",
                    "params": {
                        "test_size": 0.2
                    }
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
    inp = {
        "model": "results/optimize/%s_%s.json" % (model, dt),

        "cv": {
            "method": "time_series",
            "params": cv_params[dt],
            "max_threads": 3,
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
        for d in args.datasets:

            # Create model config
            models_fn[m](d)

            create_opt_inputs(m, d)
            create_cv_inputs(m, d)
            create_pred_inputs(m, d)
            create_eval_inputs(m, d)
            create_fit_inputs(m, d)
