import time
import numpy as np
from datetime import datetime, timedelta
from pydl.datasets import mackey_glass, get_stock_historical_data, get_log_return, create_dataset


def sp500_data_set():
    today = time.strftime("%Y-%m-%d")
    sp500 = get_stock_historical_data('^GSPC', '2000-01-01', today, usecols=['Close'])
    sp500_log_ret = get_log_return(sp500['Close'])

    # split into train and test sets
    test_start = (datetime.now() + timedelta(days=-365)).strftime("%Y-%m-%d")
    train = sp500_log_ret[:test_start]
    test = sp500_log_ret[test_start:]

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    look_back = 15
    look_ahead = 1
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    np.save('data/sp500_train_x.npy', x_train)
    np.save('data/sp500_train_y.npy', y_train)
    np.save('data/sp500_test_x.npy', x_test)
    np.save('data/sp500_test_y.npy', y_test)


def mg_data_set():
    look_back = 10
    look_ahead = 1

    mg = mackey_glass(sample_len=8020, seed=42)

    # split into train and test sets
    train = mg[:(7500+look_back)]
    test = mg[(7500+look_back):]

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    np.save('data/mg_train_x.npy', x_train)
    np.save('data/mg_train_y.npy', y_train)
    np.save('data/mg_test_x.npy', x_test)
    np.save('data/mg_test_y.npy', y_test)

if __name__ == '__main__':
    sp500_data_set()
    mg_data_set()
