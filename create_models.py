from pydl.hyperopt import *
from pydl.models.utils import save_json


def create_lstm():
    for d in ['sp500', 'mg']:
        name = 'lstm_%s' % d
        space = hp_space({
            'model': {
                'class_name': 'RNN',
                'config': hp_choice([
                    {
                        'name': name,
                        'cell_type': 'lstm',
                        'stateful': False,
                        'layers': [hp_int(10, 512)],
                        'dropout': hp_float(0, 0.5),
                        'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                        'num_epochs': hp_int(100, 500),
                        'batch_size': hp_int(32, 512),
                        'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                        'learning_rate': hp_float(0.0001, 0.01)
                    },
                    {
                        'name': name,
                        'cell_type': 'lstm',
                        'stateful': False,
                        'layers': [hp_int(10, 512), hp_int(10, 512)],
                        'dropout': [hp_float(0, 0.5), hp_float(0, 0.5)],
                        'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                        'num_epochs': hp_int(100, 500),
                        'batch_size': hp_int(32, 512),
                        'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                        'learning_rate': hp_float(0.0001, 0.01)
                    },
                    {
                        'name': name,
                        'cell_type': 'lstm',
                        'stateful': False,
                        'layers': [hp_int(10, 512), hp_int(10, 512), hp_int(10, 512)],
                        'dropout': [hp_float(0, 0.5), hp_float(0, 0.5), hp_float(0, 0.5)],
                        'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                        'num_epochs': hp_int(100, 500),
                        'batch_size': hp_int(32, 512),
                        'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                        'learning_rate': hp_float(0.0001, 0.01)
                    }
                ])
            }
        })

        save_json(space.to_json(), 'models/%s.json' % name)


def create_sae():
    pass


def create_sdae():
    pass


def create_mlp():
    for d in ['sp500', 'mg']:
        name = 'mlp_%s' % d
        space = hp_space({
            'model': {
                'class_name': 'MLP',
                'config': hp_choice([
                    {
                        'name': name,
                        'layers': [hp_int(10, 512)],
                        'dropout': hp_float(0, 0.5),
                        'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                        'l1_reg': hp_float(0, 0.001),
                        'l2_reg': hp_float(0, 0.001),
                        'num_epochs': hp_int(100, 500),
                        'batch_size': hp_int(32, 512),
                        'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                        'learning_rate': hp_float(0.0001, 0.01)
                    },
                    {
                        'name': name,
                        'layers': [hp_int(10, 512), hp_int(10, 512)],
                        'dropout': [hp_float(0, 0.5), hp_float(0, 0.5)],
                        'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                        'l1_reg': hp_float(0, 0.001),
                        'l2_reg': hp_float(0, 0.001),
                        'num_epochs': hp_int(100, 500),
                        'batch_size': hp_int(32, 512),
                        'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                        'learning_rate': hp_float(0.0001, 0.01)
                    },
                    {
                        'name': name,
                        'layers': [hp_int(10, 512), hp_int(10, 512), hp_int(10, 512)],
                        'dropout': [hp_float(0, 0.5), hp_float(0, 0.5), hp_float(0, 0.5)],
                        'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                        'l1_reg': hp_float(0, 0.001),
                        'l2_reg': hp_float(0, 0.001),
                        'num_epochs': hp_int(100, 500),
                        'batch_size': hp_int(32, 512),
                        'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                        'learning_rate': hp_float(0.0001, 0.01)
                    }
                ])
            }
        })

        save_json(space.to_json(), 'models/%s.json' % name)


if __name__ == '__main__':
    create_lstm()
    create_sae()
    create_sdae()
    create_mlp()
