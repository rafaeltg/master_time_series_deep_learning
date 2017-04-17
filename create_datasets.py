#!/usr/bin/env python3.5

import time
import calendar
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta
from pydl.datasets import mackey_glass, get_stock_historical_data, get_log_return, create_dataset, load_csv


def sp500_data_set():
    today = time.strftime("%Y-%m-%d")
    sp500 = get_stock_historical_data('^GSPC', '2000-01-01', today, usecols=['Close'])
    sp500_log_ret = get_log_return(sp500['Close'])

    # reshape into X=[X(t-look_back), ..., X(t)] and Y=[X(t+1), ..., X(t+look_ahead)]
    look_back = 20
    look_ahead = 1
    x, y = create_dataset(sp500_log_ret, look_back, look_ahead)

    x = pd.DataFrame(data=x, index=sp500_log_ret.index.get_values()[(look_back-1):(-look_ahead)])
    y = pd.DataFrame(data=y, index=sp500_log_ret.index.get_values()[(look_back-1):(-look_ahead)])

    # split into train and test sets
    test_start = (datetime.now() + timedelta(days=-365)).strftime("%Y-%m-%d")
    x_train, y_train = x[:test_start], y[:test_start]
    x_test, y_test = x[test_start:], y[test_start:]

    np.save('data/sp500_train_x.npy', x_train.values)
    np.save('data/sp500_train_y.npy', y_train.values)
    np.save('data/sp500_test_x.npy', x_test.values)
    np.save('data/sp500_test_y.npy', y_test.values)


def mg_data_set():
    look_back = 10
    look_ahead = 1

    mg = mackey_glass(sample_len=6000 + look_back, seed=42)

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x, y = create_dataset(mg, look_back, look_ahead)

    # split into train and test sets
    x_train, y_train = x[:5500], y[:5500]
    x_test, y_test = x[5500:], y[5500:]

    np.save('data/mg_train_x.npy', x_train)
    np.save('data/mg_train_y.npy', y_train)
    np.save('data/mg_test_x.npy', x_test)
    np.save('data/mg_test_y.npy', y_test)


def energy_data_set():
    data = load_csv(filename='data_sets/2015_2016_iso_ne_ca_hourly.csv',
                    has_header=True,
                    index_col='Date',
                    dtype={'Date': datetime, 'DryBulb': np.float64, 'DewPnt': np.float64, 'Demand': np.float64})

    # Add Date-time features
    dt_feats = get_date_time_features(data.index.get_values())
    data = data.assign(**dt_feats)

    # Add temperature features
    temp_feats = get_temperature_features(data.DryBulb, data.DewPnt)
    data = data.assign(**temp_feats)
    data.drop(['DryBulb', 'DewPnt'], axis=1, inplace=True)

    # Transform demand data
    data.Demand = get_log_return(data['Demand'], periods=1)

    # Add lagged-values
    data = data.assign(Demand_24=data.Demand.shift(24),
                       Demand_168=data.Demand.shift(168))

    # Add Y variable
    data = data.assign(Y=data.Demand.shift(-1))

    # Remove rows with NA values
    data.dropna(inplace=True)

    # split into train and test sets
    x = data[data.columns.difference(['Y'])]
    test_start = pd.to_datetime('2016-09-01')
    x_train, y_train = x[:test_start], data.Y[:test_start]
    x_test, y_test = x[test_start:], data.Y[test_start:]

    np.save('data/energy_train_x.npy', x_train.values)
    np.save('data/energy_train_y.npy', y_train.values)
    np.save('data/energy_test_x.npy', x_test.values)
    np.save('data/energy_test_y.npy', y_test.values)


def get_date_time_features(dates):
    """
    Features:
    
    hour: hour of the day. (h - 1) / 23
    day: day of the month. (d - 1) / (days in month - 1)
    month: month of the year. (m - 1) / 11
    week: week of the year. (w - 1) / (weeks in year - 1)
    weekend_holiday: whether or not the date is a US federal holiday or a weekend day
    """

    hour = []
    day = []
    month = []
    week = []
    weekend_holiday = []

    start_date = pd.to_datetime(str(dates[0])).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(str(dates[-1])).strftime('%Y-%m-%d')
    us_holidays = USFederalHolidayCalendar().holidays(start=start_date, end=end_date).to_pydatetime()

    for d in dates:
        dt = pd.to_datetime(str(d)).replace(tzinfo=None).to_pydatetime()
        hour.append(dt.hour/23)
        day.append((dt.day - 1) / (calendar.monthrange(dt.year, dt.month)[1] - 1))
        month.append((dt.month - 1) / 11)
        week.append((dt.isocalendar()[1]-1) / (datetime(dt.year, 12, 28).isocalendar()[1] - 1))
        weekend_holiday.append(int((dt in us_holidays) or (dt.isocalendar()[2] in [5, 6, 7])))

    return {
        'Hour': hour,
        'Day': day,
        'Month': month,
        'Week': week,
        'Weekend_Holiday': weekend_holiday
    }


def get_temperature_features(dry_bulb, dew_pnt):
    dry_bulb_ema_24 = dry_bulb.ewm(span=24).mean().replace(0, 1e-5)
    rel_dry_bulb_24 = dry_bulb / dry_bulb_ema_24.shift(1)
    dry_bulb_ma_168 = dry_bulb.rolling(window=168).mean().replace(0, 1e-5)
    rel_dry_bulb_168 = dry_bulb / dry_bulb_ma_168.shift(1)

    dew_pnt_ema_24 = dew_pnt.ewm(span=24).mean().replace(0, 1e-5)
    rel_dew_pnt_24 = dew_pnt / dew_pnt_ema_24.shift(1)
    dew_pnt_ma_168 = dew_pnt.rolling(window=168).mean().replace(0, 1e-5)
    rel_dew_pnt_168 = dew_pnt / dew_pnt_ma_168.shift(1)

    return {
        'DryBulb_EMA_24': remove_outliers(dry_bulb_ema_24),
        'RelativeDryBulb_24': remove_outliers(rel_dry_bulb_24),
        'DryBulb_MA_168': remove_outliers(dry_bulb_ma_168),
        'RelativeDryBulb_168': remove_outliers(rel_dry_bulb_168),
        'DewPnt_EMA_24': remove_outliers(dew_pnt_ema_24),
        'RelativeDewPnt_24': remove_outliers(rel_dew_pnt_24),
        'DewPnt_MA_168': remove_outliers(dew_pnt_ma_168),
        'RelativeDewPnt_168': remove_outliers(rel_dew_pnt_168)
    }


def remove_outliers(x):
    m = x.mean()
    std = x.std()
    return x.clip(m-2.5*std, m+2.5*std)


if __name__ == '__main__':
    sp500_data_set()
    mg_data_set()
    energy_data_set()
