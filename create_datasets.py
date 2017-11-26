import os
import calendar
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
from pyts import mackey_glass, lorenz, log_ret, create_dataset, load_csv, test_stationarity, correlated_lags


def create_mg():

    """
   - Data: Mackey-Glass time-series (tau = 17, delta_t = 10)
   - Sample size: 5000 points
   - Train size: 80%
   - Test size: 20%
   - Forecast window: 1 step ahead

   """

    input_file = 'data_sets/mg.csv'
    sample_len = 5000
    look_ahead = 1
    train_size = .8

    if os.path.exists(input_file):
        mg = load_csv(
            filename=input_file,
            dtype={'Value': np.float64},
            index_col='Idx',
        )
    else:
        mg = mackey_glass(n=sample_len, seed=42)
        mg.to_csv(input_file, index_label='Idx')

    # Describe input data
    describe_data(mg, 'mg')

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x, y = create_lag_features(mg, look_ahead)

    # split into train and test sets
    train_idx = int(len(x) * train_size)
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_test, y_test = x[train_idx:], y[train_idx:]

    np.save('data/mg_train_x.npy', x_train)
    np.save('data/mg_train_y.npy', y_train)
    np.save('data/mg_test_x.npy', x_test)
    np.save('data/mg_test_y.npy', y_test)
    np.save('data/mg_test_y_index.npy', range(len(x_test)))


def create_lorenz():

    """
   - Data: X component of Lorenz time-series (sigma=10, rho=28, beta=8/3)
   - Sample size: 8000 points
   - Train size: 80%
   - Test size: 20%
   - Forecast window: 1 step ahead

   """

    input_file = 'data_sets/lorenz.csv'
    sample_len = 8000
    look_ahead = 1
    train_size = .8

    if os.path.exists(input_file):
        lrz = load_csv(
            filename=input_file,
            dtype={'X': np.float64},
            index_col='Idx',
        )
    else:
        lrz = lorenz(n=sample_len)['X']
        lrz = pd.Series(np.squeeze(MinMaxScaler().fit_transform(lrz.values.reshape(-1, 1))), index=lrz.index, name='X')
        lrz.to_csv(input_file, header=True, index_label='Idx')

    # Describe input data
    describe_data(lrz, 'lorenz')

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x, y = create_lag_features(lrz, look_ahead)

    # split into train and test sets
    train_idx = int(len(x) * train_size)
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_test, y_test = x[train_idx:], y[train_idx:]

    np.save('data/lorenz_train_x.npy', x_train)
    np.save('data/lorenz_train_y.npy', y_train)
    np.save('data/lorenz_test_x.npy', x_test)
    np.save('data/lorenz_test_y.npy', y_test)
    np.save('data/lorenz_test_y_index.npy', range(len(x_test)))


def create_energy():

    """
    - Data: ISO New England hourly energy demand from 01-01-2015 to 31-12-2016
    - Sample size: 17544
    - Train size: from 04-01-2015 to 30-11-2016 (16727)
    - Test size: from 01-12-2016 to 31-12-2016 (743)
    - Forecast window: 1 step ahead

    """

    data = load_csv(filename='data_sets/2015_2016_iso_ne_ca_hourly.csv',
                    has_header=True,
                    index_col='Date',
                    dtype={'DryBulb': np.float64, 'DewPnt': np.float64, 'Demand': np.float64})

    # Transform demand data
    demand = log_ret(data['Demand'], periods=1)

    # Describe input data
    describe_data(demand, 'energy')

    # reshape into X=[X(t-look_back), ..., X(t)] and Y=[X(t+1), ..., X(t+look_ahead)]
    x, y = create_lag_features(demand)

    # Calculate Date-time features
    time_feats = get_date_time_features(x.index.get_values())
    x = pd.concat([x, time_feats], axis=1, join_axes=[x.index])

    # split into train and test sets
    train_end = '2016-11-30T23:59:59'
    test_start = '2016-12-01'
    x_train, y_train = x[:train_end], y[:train_end]
    x_test, y_test = x[test_start:], y[test_start:]

    np.save('data/energy_train_x.npy', x_train.values)
    np.save('data/energy_train_y.npy', y_train.values)
    np.save('data/energy_test_x.npy', x_test.values)
    np.save('data/energy_test_y.npy', y_test.values)
    np.save('data/energy_test_y_index.npy', y_test.index.get_values())


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

    return pd.DataFrame(
        {
            'Hour': hour,
            'Day': day,
            'Month': month,
            'Week': week,
            'Weekend_Holiday': weekend_holiday
        },
        index=dates)


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


def create_lag_features(ts, look_ahead=1):

    """
    Features:
        - 15 most correlated lags

    :param ts:
    :param look_ahead:
    :return:
    """

    corr_lags = correlated_lags(ts, corr_lags=15, max_lags=200)
    if len(corr_lags) < 5:
        corr_lags = range(1, 21)

    look_back = corr_lags[-1]

    x, y = create_dataset(ts, look_back, look_ahead)

    # use only the most correlated lags
    corr_lags = np.negative(corr_lags)
    x = np.array([X[corr_lags] for X in x])

    x = pd.DataFrame(data=x,
                     index=ts.index.get_values()[(look_back-1):(-look_ahead)],
                     columns=map(lambda l: 'lag.%d' % -l, corr_lags))
    y = pd.DataFrame(data=y,
                     index=ts.index.get_values()[(look_back-1):(-look_ahead)])
    return x, y


def describe_data(ts, dataset):
    test_stationarity(ts.as_matrix(), 'results/desc/%s_stat.csv' % dataset)
    ts.describe().to_csv('results/desc/%s_desc.csv' % dataset)


dt_fn = {
    'mg': create_mg,
    'lorenz': create_lorenz,
    'energy': create_energy
}


def fn(x):
    dt_fn[x]()


def create_datasets(dts):
    Parallel(n_jobs=len(dts))(delayed(fn, check_pickle=False)(dt) for dt in dts)
