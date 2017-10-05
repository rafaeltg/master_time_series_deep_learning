import os
import calendar
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta
from pydl.datasets import mackey_glass, get_stock_historical_data, get_log_return, create_dataset, load_csv, acf, test_stationarity


def create_mg():

    """

   - Data: Mackey-Glass time-series (tau = 17, delta_t = 10)
   - Sample size: 4000 points
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
        mg = mackey_glass(sample_len=sample_len, seed=42)
        mg.to_csv(input_file, index_label='Idx')

    # Describe input data
    describe_data(mg, 'mg')

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x, y = create_features(mg, look_ahead)

    # split into train and test sets
    train_idx = int(len(x) * train_size)
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_test, y_test = x[train_idx:len(x)], y[train_idx:len(y)]

    np.save('data/mg_train_x.npy', x_train)
    np.save('data/mg_train_y.npy', y_train)
    np.save('data/mg_test_x.npy', x_test)
    np.save('data/mg_test_y.npy', y_test)
    np.save('data/mg_test_y_index.npy', range(len(x_test)))


def create_sp500():

    """

    - Data: Daily log return of S&P500 closing value
    - Sample size: from 2000-01-01 to 2017-07-01
    - Train size: from 2000-01-01 to 2016-06-30
    - Test size: 1 year (from 2016-07-01 to 2017-07-01)
    - Forecast window: 1 step ahead

    """

    input_file = 'data_sets/sp500_daily_log_return.csv'
    end = "2017-07-01"

    if os.path.exists(input_file):
        sp500_log_ret = load_csv(
            filename=input_file,
            dtype={'Close': np.float64},
            index_col='Date',
        )
    else:
        sp500 = get_stock_historical_data('SPY', '2000-01-01', end, usecols=['Close'])
        sp500_log_ret = get_log_return(sp500['Close'])
        sp500_log_ret.to_csv(input_file, date_format='%m-%d-%Y', index_label='Date')

    # Describe input data
    describe_data(sp500_log_ret, 'sp500')

    # reshape into X=[X(t-look_back), ..., X(t)] and Y=[X(t+1), ..., X(t+look_ahead)]
    x, y = create_features(sp500_log_ret)

    # split into train and test sets
    test_start = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=-365)).strftime("%Y-%m-%d")
    x_train, y_train = x[:test_start], y[:test_start]
    x_test, y_test = x[test_start:], y[test_start:]

    np.save('data/sp500_train_x.npy', x_train.values)
    np.save('data/sp500_train_y.npy', y_train.values)
    np.save('data/sp500_test_x.npy', x_test.values)
    np.save('data/sp500_test_y.npy', y_test.values)
    np.save('data/sp500_test_y_index.npy', y_test.index.get_values())


def create_energy():

    """

   - Data:
   - Sample size:
   - Train size:
   - Test size:
   - Forecast window: 1 step ahead

   """

    data = load_csv(filename='data_sets/2015_2016_iso_ne_ca_hourly.csv',
                    has_header=True,
                    index_col='Date',
                    dtype={'Date': datetime, 'DryBulb': np.float64, 'DewPnt': np.float64, 'Demand': np.float64})

    # Describe input data
    describe_data(data, 'energy')

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


def create_features(ts, look_ahead=1):

    """
    Features:
        - 10 most correlated lags

    :param ts:
    :param look_ahead:
    :return:
    """

    acfs, conf = acf(ts, 100)
    acfs = np.asarray(acfs[1:])

    most_corr = [v > conf for v in acfs]
    if sum(most_corr) >= 1:
        corr_lags = np.argsort(acfs[most_corr])[-10:]
        corr_lags = sorted(corr_lags)
        look_back = corr_lags[-1] + 1
    else:
        look_back = 20
        corr_lags = range(look_back)

    x, y = create_dataset(ts, look_back, look_ahead)

    # use only the most correlated lags
    x = np.array([X[corr_lags] for X in x])

    x = pd.DataFrame(data=x, index=ts.index.get_values()[(look_back-1):(-look_ahead)])
    y = pd.DataFrame(data=y, index=ts.index.get_values()[(look_back-1):(-look_ahead)])
    return x, y


def describe_data(ts, dataset):
    test_stationarity(ts.as_matrix(), 'results/desc/%s_stat.csv' % dataset)
    ts.describe().to_csv('results/desc/%s_desc.csv' % dataset)


dt_fn = {
    'mg': create_mg,
    'sp500': create_sp500,
    'energy': create_energy
}


def fn(x):
    dt_fn[x]()


def create_datasets(dts):
    Parallel(n_jobs=len(dts))(delayed(fn, check_pickle=False)(dt) for dt in dts)
