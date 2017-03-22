from pydl.models.utils import save_json, load_json


cv_params = {
    'sp500': {
        "window": 2772,
        "horizon": 252,
        "fixed": False,
        "by": 252
    },

    'mg': {
        "window": 6000,
        "horizon": 500,
        "fixed": False,
        "by": 500
    }
}


def create_opt_inputs():

    for m in ['lstm', 'mlp']:
        for d in ['sp500', 'mg']:
            inp = {
                "hp_space": load_json('models/%s_%s.json' % (m, d)),

                "opt": {
                    "method": "cmaes",
                    "params": {
                        "pop_size": 5,
                        "max_iter": 2
                    },

                    "obj_fn": {
                        "cv": {
                            "method": "time_series",
                            "params": cv_params[d]
                        }
                    }
                },

                'data_set': {
                    "data_x": {
                        "path": "data/%s_train_x.csv" % d,
                        "params": {
                            "has_header": False
                        }
                    },
                    "data_y": {
                        "path": "data/%s_train_y.csv" % d,
                        "params": {
                            "has_header": False
                        }
                    }
                }
            }
            save_json(inp, 'inputs/%s_%s_opt.json' % (m, d))


def create_cv_inputs():
    pass


def create_pred_inputs():
    pass


if __name__ == '__main__':
    create_opt_inputs()
    create_cv_inputs()
    create_pred_inputs()
