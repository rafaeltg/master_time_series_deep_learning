import pandas as pd
from pydl.datasets import mackey_glass, get_stock_historical_data, get_log_return, create_dataset


def sp500_data_set():
    sp500 = get_stock_historical_data('^GSPC', '2000-01-01', '2017-03-21', usecols=['Close'])
    sp500_log_ret = get_log_return(sp500['Close'])

    # split into train and test sets
    train = sp500_log_ret[:'2016-03-21']
    test = sp500_log_ret['2016-03-21':]

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    look_back = 15
    look_ahead = 1
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    pd.DataFrame(x_train).to_csv('data/sp500_train_x.csv', header=False, index=False)
    pd.DataFrame(y_train).to_csv('data/sp500_train_y.csv', header=False, index=False)
    pd.DataFrame(x_test).to_csv('data/sp500_test_x.csv', header=False, index=False)
    pd.DataFrame(y_test).to_csv('data/sp500_test_y.csv', header=False, index=False)


def mg_data_set():

    look_back = 10
    look_ahead = 1

    mg = mackey_glass(sample_len=9020, seed=42)

    # split into train and test sets
    train = mg[:(8000+look_back)]
    test = mg[(8000+look_back):]

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    pd.DataFrame(x_train).to_csv('data/mg_train_x.csv', header=False, index=False)
    pd.DataFrame(y_train).to_csv('data/mg_train_y.csv', header=False, index=False)
    pd.DataFrame(x_test).to_csv('data/mg_test_x.csv', header=False, index=False)
    pd.DataFrame(y_test).to_csv('data/mg_test_y.csv', header=False, index=False)

if __name__ == '__main__':
    sp500_data_set()
    mg_data_set()
