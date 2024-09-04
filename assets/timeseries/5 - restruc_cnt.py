import numpy as np, pandas as pd, preproc as pp, split_sample as ss

def restruc_cnt(df, config, cnt_var, test_days=0, future_days=1):
    """
    Pre-process the data and restructure the continuous variables, so it can be read by tensorflow.

    - cnt_var = list of continuous variable names
    - test_days = number of test days; default = 0
    - future_days = number of future days to predict; default = 1
    - n_days = number of prior days to include in the model predictors; 
    """
    n_days, _, _, _, _, _, _ = config
    df = df[(['Date', 'Revenue'] + cnt_var)]
    trn_data, tst_data, y_scaler, x_scaler, trn_dates = ss.split_sample(processed_data=df,
      test_days=test_days,
      cnt_var=cnt_var)
    iv_num = trn_data.shape[1]
    depvar = [
     'var1(t)']
    for i in range(1, future_days):
        value = 'var1(t+' + str(i) + ')'
        depvar.append(value)

    ivar = []
    for i in range(2, iv_num + 1):
        value = 'var' + str(i) + '(t-1)'
        ivar.append(value)

    ivarr = []
    for i in range(1, int(n_days) + 1):
        value = 'var1(t-' + str(i) + ')'
        ivarr.append(value)

    print('Restructuring the continuous data...\n')
    trn_data_reframe = pp.series_to_supervised(trn_data, int(n_days), future_days)
    tst_data_reframe = pp.series_to_supervised(tst_data, int(n_days), future_days)
    trn_data_reframe_X = trn_data_reframe[(ivarr + ivar)]
    tst_data_reframe_X = tst_data_reframe[(ivarr + ivar)]
    trn_data_reframe_y = trn_data_reframe[depvar]
    tst_data_reframe_y = tst_data_reframe[depvar]
    trn_X = trn_data_reframe_X.values
    tst_X = tst_data_reframe_X.values
    trn_y = trn_data_reframe_y.values
    tst_y = tst_data_reframe_y.values
    trn_X_cnt = trn_X.reshape((trn_X.shape[0], 1, trn_X.shape[1]))
    tst_X_cnt = tst_X.reshape((tst_X.shape[0], 1, tst_X.shape[1]))
    print('FYI:')
    print('\t- After restructuring, the continuous data shape is ', trn_X_cnt.shape)
    print('\nThe continuous data is now ready for modeling...\n')
    return (
     trn_X_cnt, trn_y, x_scaler, y_scaler, tst_X_cnt, tst_y, trn_dates, ivar, ivarr)