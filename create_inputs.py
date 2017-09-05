#!/usr/bin/env python3.5

from pydl.models.utils import save_json, load_json


models = ['lstm', 'sae', 'sdae', 'mlp']

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
        "window": 7247,
        "horizon": 1440,
        "fixed": False,
        "by": 1440
    }
}


def create_opt_inputs():
    for m in models:
        for d in cv_params.keys():
            inp = {
                "hp_space": load_json('models/%s_%s.json' % (m, d)),

                "opt": {
                    "method": {
                        "class": "cmaes",
                        "params": {
                            "pop_size": 24,
                            "max_iter": 50
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
                        "path": "data/%s_train_x.npy" % d,
                    },
                    "data_y": {
                        "path": "data/%s_train_y.npy" % d,
                    }
                }
            }
            save_json(inp, 'inputs/%s_%s_optimize.json' % (m, d))


def create_fit_inputs():
    for m in models:
        for d in cv_params.keys():
            inp = {
                'model': "results/optimize/%s_%s.json" % (m, d),

                'data_set': {
                    'train_x': {
                        'path': "data/%s_train_x.npy" % d,
                    },
                    'train_y': {
                        'path': "data/%s_train_y.npy" % d,
                    }
                }

            }
            save_json(inp, 'inputs/%s_%s_fit.json' % (m, d))


def create_cv_inputs():
    for m in models:
        for d in cv_params.keys():
            inp = {
                "model": "results/optimize/%s_%s.json" % (m, d),

                "cv": {
                    "method": "time_series",
                    "params": cv_params[d],
                    "max_threads": 3
                },

                'data_set': {
                    "data_x": {
                        "path": "data/%s_train_x.npy" % d,
                    },
                    "data_y": {
                        "path": "data/%s_train_y.npy" % d,
                    }
                }
            }
            save_json(inp, 'inputs/%s_%s_cv.json' % (m, d))


def create_pred_inputs():
    for m in models:
        for d in cv_params.keys():
            inp = {
                "model": "results/fit/%s_%s.json" % (m, d),

                'data_set': {
                    "data_x": {
                        "path": "data/%s_test_x.npy" % d,
                    }
                }
            }
            save_json(inp, 'inputs/%s_%s_predict.json' % (m, d))


def create_eval_inputs():
    for m in models:
        for d in cv_params.keys():
            inp = {
                "model": "results/fit/%s_%s.json" % (m, d),

                'data_set': {
                    "data_x": {
                        "path": "data/%s_test_x.npy" % d,
                    },
                    "data_y": {
                        "path": "data/%s_test_y.npy" % d,
                    }
                },

                'scoring': ['rmse', 'mse', 'mae', 'r2'],

                'errors_acf': {
                    "nlags": 30
                }
            }

            save_json(inp, 'inputs/%s_%s_eval.json' % (m, d))


if __name__ == '__main__':
    create_opt_inputs()
    create_cv_inputs()
    create_pred_inputs()
    create_eval_inputs()
    create_fit_inputs()
