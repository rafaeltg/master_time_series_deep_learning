from pydl.hyperopt import *
from pydl.models.utils import save_json


model_params = {
    'num_epochs': hp_int(100, 500),
    'batch_size': hp_int(32, 512),
    'opt': hp_choice(['rmsprop', 'adagrad', 'adadelta', 'adam']),
    'learning_rate': hp_float(0.0001, 0.01)
}


def add_model_params(p={}):
    p.update(dict(model_params.items()))
    return p


def supervised_model_params(name, layers, p={}):
    p = add_model_params(p)
    p.update(dict(
        name=name,
        layers=layers,
        activation=hp_choice(['relu', 'tanh', 'sigmoid', 'linear']),
        dropout=[hp_float(0, 0.5)] * len(layers),
    ))
    return p


def create_lstm():

    def lstm(name, layers):
        return supervised_model_params(
            name=name,
            layers=layers,
            p={
                'cell_type': 'lstm',
                'stateful': False,
            },
        )

    for d in ['sp500', 'mg']:
        name = 'lstm_%s' % d
        space = hp_space({
            'model': {
                'class_name': 'RNN',
                'config': hp_choice([
                    lstm(name, layers=[hp_int(10, 512)]),
                    lstm(name, layers=[hp_int(10, 512), hp_int(10, 512)]),
                    lstm(name, layers=[hp_int(10, 512), hp_int(10, 512), hp_int(10, 512)])
                ])
            }
        })

        save_json(space.to_json(), 'models/%s.json' % name)


def create_sae():
    ae = {
        'class_name': 'Autoencoder',
        'config': add_model_params({
            'n_hidden': hp_int(10, 100),
            'enc_activation': hp_choice(['relu', 'tanh', 'sigmoid', 'linear']),
            'l1_reg': hp_float(0, 0.001),
            'l2_reg': hp_float(0, 0.001)
        })
    }

    for d in ['sp500', 'mg']:
        name = 'sae_%s' % d
        space = hp_space({
            'model': {
                'class_name': 'StackedAutoencoder',
                'config': hp_choice([
                    supervised_model_params(name=name, layers=[ae]),
                    supervised_model_params(name=name, layers=[ae, ae]),
                    supervised_model_params(name=name, layers=[ae, ae, ae])
                ])
            }
        })

        save_json(space.to_json(), 'models/%s.json' % name)


def create_sdae():
    dae = {
        'class_name': 'DenoisingAutoencoder',
        'config': add_model_params({
            'n_hidden': hp_int(10, 100),
            'enc_activation': hp_choice(['relu', 'tanh', 'sigmoid', 'linear']),
            'l1_reg': hp_float(0, 0.001),
            'l2_reg': hp_float(0, 0.001),
            'corr_type': hp_choice(['gaussian', 'masking']),
            'corr_param': hp_float(1e-5, 1)
        })
    }

    for d in ['sp500', 'mg']:
        name = 'sdae_%s' % d
        space = hp_space({
            'model': {
                'class_name': 'StackedAutoencoder',
                'config': hp_choice([
                    supervised_model_params(name=name, layers=[dae]),
                    supervised_model_params(name=name, layers=[dae, dae]),
                    supervised_model_params(name=name, layers=[dae, dae, dae])
                ])
            }
        })

        save_json(space.to_json(), 'models/%s.json' % name)


def create_mlp():

    def mlp(name, layers):
        return supervised_model_params(
            name=name,
            layers=layers,
            p={
                'l1_reg': hp_float(0, 0.001),
                'l2_reg': hp_float(0, 0.001)
            }
        )

    for d in ['sp500', 'mg']:
        name = 'mlp_%s' % d
        space = hp_space({
            'model': {
                'class_name': 'MLP',
                'config': hp_choice([
                    mlp(name, layers=[hp_int(10, 512)]),
                    mlp(name, layers=[hp_int(10, 512), hp_int(10, 512)]),
                    mlp(name, layers=[hp_int(10, 512), hp_int(10, 512), hp_int(10, 512)])
                ])
            }
        })

        save_json(space.to_json(), 'models/%s.json' % name)


if __name__ == '__main__':
    create_lstm()
    create_sae()
    create_sdae()
    create_mlp()