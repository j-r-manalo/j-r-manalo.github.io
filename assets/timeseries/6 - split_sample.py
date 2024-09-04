from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta, datetime
import numpy as np

def split_sample(processed_data, test_days, cnt_var):
    """Splits file into training and testing samples"""
    traincutoff = max(processed_data['Date']) - timedelta(test_days - 1)
    testcutoff = max(processed_data['Date']) - timedelta(test_days - 2)
    trn_size = len(processed_data) - test_days
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    print('Pre-process: Normalizing data...\n')
    col_names = cnt_var
    trn_dates = processed_data['Date']
    trn_dates = trn_dates[0:trn_size - 1,]
    y = processed_data['Revenue'].values
    y = y.reshape(-1, 1)
    y = y_scaler.fit_transform(y)
    processed_data = x_scaler.fit_transform(processed_data[col_names])
    processed_data = np.append(y, processed_data, axis=1)
    print('Pre-process: Splitting the data...\n')
    print('Cutting off training data at', traincutoff, ', which is', test_days, 'days from the capped date of the file.')
    print('Starting test data at', testcutoff, ', which includes', test_days, ' number of days.\n')
    trn_data, tst_data = processed_data[0:trn_size, :], processed_data[trn_size:len(processed_data), :]
    print('FYI:')
    print('\t- Full data shape is ', processed_data.shape)
    print('\t- Training input data shape is ', trn_data.shape)
    print('\t- Test input data shape is ', tst_data.shape)
    print(' ')
    return (
     trn_data, tst_data, y_scaler, x_scaler, trn_dates)