from pydl.hyperopt import *
from pydl.models.utils import save_json


data_sets = ["sp500", "mg", "energy"]

model_params = {
    'nb_epochs': hp_int(50, 500),
    'batch_size': hp_int(32, 1024),
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
        early_stopping=True,
        patient=hp_int(1, 40),
        min_delta=hp_float(1e-5, 1e-4)
    ))
    return p


def create_lstm(dataset, save=True):

    def lstm(name, layers):
        return supervised_model_params(
            name=name,
            layers=layers,
            p={
                'cell_type': 'lstm',
                'stateful': False,
            },
        )

    name = 'lstm_%s' % dataset
    lstm_space = hp_space({
        'model': {
            'class_name': 'RNN',
            'config': hp_choice([
                lstm(name, layers=[hp_int(8, 512)]),
                lstm(name, layers=[hp_int(8, 512), hp_int(8, 512)]),
                lstm(name, layers=[hp_int(8, 512), hp_int(8, 512), hp_int(8, 512)])
            ])
        }
    })

    if save:
        save_json(lstm_space.to_json(), 'models/%s.json' % name)

    return lstm_space


def create_sae(dataset, save=True):
    ae = {
        'class_name': 'Autoencoder',
        'config': add_model_params({
            'n_hidden': hp_int(8, 512),
            'enc_activation': hp_choice(['relu', 'tanh', 'sigmoid', 'linear']),
            'l1_reg': hp_float(0, 0.001),
            'l2_reg': hp_float(0, 0.001)
        })
    }

    name = 'sae_%s' % dataset
    sae_space = hp_space({
        'model': {
            'class_name': 'StackedAutoencoder',
            'config': hp_choice([
                supervised_model_params(name=name, layers=[ae]),
                supervised_model_params(name=name, layers=[ae, ae]),
                supervised_model_params(name=name, layers=[ae, ae, ae])
            ])
        }
    })

    if save:
        save_json(sae_space.to_json(), 'models/%s.json' % name)

    return sae_space


def create_sdae(dataset, save=True):
    dae = {
        'class_name': 'DenoisingAutoencoder',
        'config': add_model_params({
            'n_hidden': hp_int(8, 512),
            'enc_activation': hp_choice(['relu', 'tanh', 'sigmoid', 'linear']),
            'l1_reg': hp_float(0, 0.001),
            'l2_reg': hp_float(0, 0.001),
            'corr_type': hp_choice(['gaussian', 'masking']),
            'corr_param': hp_float(1e-5, 1)
        })
    }

    name = 'sdae_%s' % dataset
    sdae_space = hp_space({
        'model': {
            'class_name': 'StackedAutoencoder',
            'config': hp_choice([
                supervised_model_params(name=name, layers=[dae]),
                supervised_model_params(name=name, layers=[dae, dae]),
                supervised_model_params(name=name, layers=[dae, dae, dae])
            ])
        }
    })

    if save:
        save_json(sdae_space.to_json(), 'models/%s.json' % name)

    return sdae_space


def create_mlp(dataset, save=True):

    def mlp(name, layers):
        return supervised_model_params(
            name=name,
            layers=layers,
            p={
                'l1_reg': hp_float(0, 0.001),
                'l2_reg': hp_float(0, 0.001)
            }
        )

    name = 'mlp_%s' % dataset
    mlp_space = hp_space({
        'model': {
            'class_name': 'MLP',
            'config': hp_choice([
                mlp(name, layers=[hp_int(8, 512)]),
                mlp(name, layers=[hp_int(8, 512), hp_int(8, 512)]),
                mlp(name, layers=[hp_int(8, 512), hp_int(8, 512), hp_int(8, 512)])
            ])
        }
    })

    if save:
        save_json(mlp_space.to_json(), 'models/%s.json' % name)

    return mlp_space


models_fn = {
    'lstm': create_lstm,
    'sae': create_sae,
    'sdae': create_sdae,
    'mlp': create_mlp
}