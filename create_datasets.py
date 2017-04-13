import time
import calendar
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta
from pydl.datasets import mackey_glass, get_stock_historical_data, get_log_return, create_dataset, load_csv, acf, test_stationarity, decompose
import matplotlib.pyplot as plt


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

    mg = mackey_glass(sample_len=6020, seed=42)

    # split into train and test sets
    train = mg[:(5500+look_back)]
    test = mg[(5500+look_back):]

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    np.save('data/mg_train_x.npy', x_train)
    np.save('data/mg_train_y.npy', y_train)
    np.save('data/mg_test_x.npy', x_test)
    np.save('data/mg_test_y.npy', y_test)


def energy_data_set():

    data = load_csv(filename='data_sets/2015_2016_iso_ne_ca_hourly.csv',
                    has_header=True,
                    index_col='Date',
                    dtype={'Date': datetime, 'DryBulb': np.float64, 'DewPnt': np.float64, 'Demand': np.float64})

    # Calculate Date-time features
    dt_feats = get_date_time_features(data.index.get_values())

    data = data.assign(**dt_feats)

    #print(data.head(24))
    #print(data.tail(24))

    #acf(data['Demand'], nlags=48, plot=True)
    #pacf(data['Demand'], nlags=48, plot=True)

    #test_stationarity(data['Demand'])

    decompose(data['Demand'].values[:1000], plot=True)

    log_ret_1 = get_log_return(data['Demand'], periods=1)
    #acf(log_ret_1['Demand'], nlags=48, plot=True)
    #test_stationarity(log_ret_1['Demand'])

    log_ret_24 = get_log_return(data['Demand'], periods=24)
    acf(log_ret_24['Demand'], nlags=48, plot=True)
    test_stationarity(log_ret_24['Demand'])

    plt.subplot(311)
    plt.plot(data['Demand'].head(800), label='Original')
    plt.subplot(312)
    plt.plot(log_ret_1.values[:799], label='log_ret_1')
    plt.subplot(313)
    plt.plot(log_ret_24.values[:(800-24)], label='log_ret_24')
    plt.show(block=True)


def get_date_time_features(dates):
    """
    Features:
    
    hour: hour of the day. (h - 1) / 23
    day: day of the month. (d - 1) / (days in month - 1)
    week_day: day of the week. (week day - 1) / 6 (Monday = 1 and Sunday = 7)
    month: month of the year. (m - 1) / 11
    week: week of the year. (w - 1) / (weeks in year - 1)
    holiday: whether or not the date is a US federal holiday
    """

    hour = []
    day = []
    week_day = []
    month = []
    week = []
    holiday = []

    start_date = pd.to_datetime(str(dates[0])).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(str(dates[-1])).strftime('%Y-%m-%d')
    us_holidays = USFederalHolidayCalendar().holidays(start=start_date, end=end_date).to_pydatetime()

    for d in dates:
        dt = pd.to_datetime(str(d)).replace(tzinfo=None).to_pydatetime()
        hour.append(dt.hour/23)
        day.append((dt.day - 1) / (calendar.monthrange(dt.year, dt.month)[1] - 1))
        week_day.append((dt.isocalendar()[2]-1)/6)
        month.append((dt.month - 1) / 11)
        week.append((dt.isocalendar()[1]-1) / (datetime(dt.year, 12, 28).isocalendar()[1] - 1))
        holiday.append(int(dt in us_holidays))

    return {
        'Hour': hour,
        'Day': day,
        'WeekDay': week_day,
        'Month': month,
        'Week': week,
        'Holiday': holiday
    }


if __name__ == '__main__':
    #sp500_data_set()
    #mg_data_set()
    energy_data_set()
