import numpy as np, pandas as pd, preproc as pp, split_sample as ss
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta, datetime
import holiday_calendar as hc
from statistics import stdev

def forecasts(df, config, model, cnt_var, x_scaler, y_scaler, ivar, ivarr, test_days=0, future_days=1):
    """
    Pre-process the data and restructure the continuous variables, so it can be read by tensorflow.

    - model = best model
    - cnt_var = list of continuous variable names
    - x_scaler = scalers for independent variables; used for normalization after modeling
    - y_scaler = scalers for dependent variable (Revenue); used for normalization after modeling 
    - ivar = list of resteructured continuous variables
    - ivarr = list of resteructured revenue fields that are pulled in as indep. vars. using prior_days variable
    - test_days = number of test days; default = 0
    - future_days = number of future days to predict; default = 1
    - n_days = number of prior days to include in the model predictors
    """
    n_days, _, _, _, _, _, _ = config
    pd.options.mode.chained_assignment = None
    df_fin_t_maxdate = df.agg({'Date': max})
    df_forecast = pd.date_range(start=(df_fin_t_maxdate[0]),
      periods=150).to_frame(index=False)
    df_forecast.columns = ['Date']
    df_forecast['year'] = df_forecast['Date'].apply(lambda x: x.year)
    df_forecast['quarter'] = df_forecast['Date'].apply(lambda x: x.quarter)
    df_forecast['month'] = df_forecast['Date'].apply(lambda x: x.month)
    df_forecast['day'] = df_forecast['Date'].apply(lambda x: x.day)
    df_forecast['weekday'] = df_forecast['Date'].apply(lambda x: x.dayofweek)
    df_forecast['weekend_flag'] = np.where(df_forecast['weekday'] >= 5, 1, 0)
    hol_cal = hc.holiday_calendar(df_forecast)
    df_forecast = pd.merge(df_forecast, hol_cal,
      left_on='Date',
      right_on='Date',
      how='left')
    df_forecast['Holiday'].fillna('Not Holiday', inplace=True)
    df_forecast['holiday_flag'] = np.where(df_forecast['Holiday'] != 'Not Holiday', 1, 0)
    df_fin_cnt_hist_dates = df_forecast['Date'] - timedelta(365)
    df_fin_cnt_hist = df[(['Date', 'Revenue'] + cnt_var)]
    df_fin_cnt_hist = df_fin_cnt_hist[((df_fin_cnt_hist['Date'] >= min(df_fin_cnt_hist_dates) - timedelta(n_days.astype(float))) & (df_fin_cnt_hist['Date'] <= max(df_fin_cnt_hist_dates)))].reset_index(drop=True)
    df_fin_cnt_pctchnge = df_fin_cnt_hist[([
     'Revenue'] + cnt_var)].pct_change().fillna(0)
    for var in ['Revenue'] + cnt_var:
        for i in range(0, len(df_fin_cnt_pctchnge)):
            if df_fin_cnt_pctchnge.loc[(i, var)] > 2:
                df_fin_cnt_pctchnge.loc[(i, var)] = 2
            elif df_fin_cnt_pctchnge.loc[(i, var)] < -2:
                df_fin_cnt_pctchnge.loc[(i, var)] = -2
            else:
                df_fin_cnt_pctchnge.loc[(i, var)] = df_fin_cnt_pctchnge.loc[(i, var)]

    df_fin_t2 = df[(['Date'] + ['Revenue'] + cnt_var)]
    df_fin_t2 = df_fin_t2[((df_fin_t2['Date'] >= max(df['Date']) - timedelta(n_days.astype(float))) & (df_fin_t2['Date'] <= max(df['Date'])))].reset_index(drop=True)
    df_fin_t2 = df_fin_t2.drop(['Date'], axis=1)
    df_cnt_est = pd.DataFrame(columns=(['Revenue'] + cnt_var))
    for var in ['Revenue'] + cnt_var:
        for i in range(0, len(df_fin_cnt_pctchnge)):
            if i <= n_days:
                df_cnt_est.loc[(i, var)] = df_fin_t2.loc[(i, var)]
            else:
                df_cnt_est.loc[(i, var)] = df_cnt_est.loc[(i - 1, var)] + df_fin_cnt_pctchnge.loc[(i, var)] * df_cnt_est.loc[(i - 1, var)]

    y_est_scaled = df_cnt_est['Revenue'].values
    y_est_scaled = y_est_scaled.reshape(-1, 1)
    y_est_scaled = y_scaler.transform(y_est_scaled)
    df_cnt_est_scaled = x_scaler.transform(df_cnt_est[cnt_var])
    df_cnt_est_scaled = np.append(y_est_scaled, df_cnt_est_scaled, axis=1)
    df_cnt_est_scaled_reframe = pp.series_to_supervised(df_cnt_est_scaled, n_days, future_days)
    df_cnt_est_scaled_reframe_X = df_cnt_est_scaled_reframe[(ivarr + ivar)]
    df_cnt_est_scaled_reframe_X.drop((df_cnt_est_scaled_reframe_X.index[0]),
      inplace=True)
    df_cnt_est_X = df_cnt_est_scaled_reframe_X.values
    df_cnt_est_X = df_cnt_est_X.reshape((
     df_cnt_est_X.shape[0], 1, df_cnt_est_X.shape[1]))
    df_cat_est = df_forecast.drop([0])
    categorical_vars = [
     'year', 'quarter', 'month',
     'weekday', 'weekend_flag', 'holiday_flag']
    df_cat_est = df_cat_est[categorical_vars]
    df_cat_est_X, _ = pp.refrac_cat(df_cat_est,
      test_days=0, categorical_vars=categorical_vars)
    df_cat_est_X_lst = pp.X_cat_lst(df_cat_est_X, future_days - 1)
    y_forecast = model.predict([df_cat_est_X_lst[0], df_cat_est_X_lst[1], df_cat_est_X_lst[2],
     df_cat_est_X_lst[3], df_cat_est_X_lst[4], df_cat_est_X_lst[5],
     df_cnt_est_X])
    y_forecast_rescale = y_scaler.inverse_transform(y_forecast)
    y_forecast_rescale = pd.DataFrame(y_forecast_rescale)
    y_forecast_rescale.columns = ['Forcasted_Revenue']
    forecast_dates = pd.DataFrame(df_forecast['Date'].drop([0])).reset_index(drop=True)
    y_forecast_rescale['Date'] = forecast_dates
    y_forecast_rescale = y_forecast_rescale.set_index('Date')
    rev_std = stdev(df_fin_cnt_hist['Revenue'])
    y_forecast_rescale_t = y_forecast_rescale.reset_index()
    y_forecast_rescale_t.columns = ['Date', 'Revenue']
    for i in range(0, len(y_forecast_rescale_t)):
        if y_forecast_rescale_t.loc[(i, 'Revenue')] >= 0:
            y_forecast_rescale_t.loc[(i, 'Revenue')] = y_forecast_rescale_t.loc[(i, 'Revenue')].astype(int)

    y_forecast_rescale_t['year'] = y_forecast_rescale_t['Date'].apply(lambda x: x.year)
    y_forecast_rescale_t['quarter'] = y_forecast_rescale_t['Date'].apply(lambda x: x.quarter)
    y_forecast_rescale_t['month'] = y_forecast_rescale_t['Date'].apply(lambda x: x.month)
    y_forecast_rescale_t['Forecasted Min'] = ''
    y_forecast_rescale_t['Forecasted Max'] = ''
    for i in range(0, len(y_forecast_rescale_t)):
        if i < 90:
            y_forecast_rescale_t.loc[(i, 'Forecasted Min')] = int(y_forecast_rescale_t.loc[(i, 'Revenue')] - 1 * rev_std - 2 * (i / 90) * rev_std)
            y_forecast_rescale_t.loc[(i, 'Forecasted Max')] = int(y_forecast_rescale_t.loc[(i, 'Revenue')] + 1 * rev_std + 2 * (i / 90) * rev_std)
        if i >= 90:
            y_forecast_rescale_t.loc[(i, 'Forecasted Min')] = int(y_forecast_rescale_t.loc[(i, 'Revenue')] - 1 * rev_std - 2 * rev_std)
            y_forecast_rescale_t.loc[(i, 'Forecasted Max')] = int(y_forecast_rescale_t.loc[(i, 'Revenue')] + 1 * rev_std + 2 * rev_std)

    df_fin_t3 = df[['Date', 'Revenue', 'year', 'quarter', 'month']]
    for i in range(0, len(df_fin_t3)):
        if df_fin_t3.loc[(i, 'Revenue')] >= 0:
            df_fin_t3.loc[(i, 'Revenue')] = df_fin_t3.loc[(i, 'Revenue')].astype(int)

    df_fin_t3['Forecasted Min'] = ''
    df_fin_t3['Forecasted Max'] = ''
    df_fin_t3['Forecasted Min'].iloc[[-1]] = df_fin_t3['Revenue'].iloc[[-1]]
    df_fin_t3['Forecasted Max'].iloc[[-1]] = df_fin_t3['Revenue'].iloc[[-1]]
    df_hist_plus_forecast = pd.concat([
     df_fin_t3, y_forecast_rescale_t],
      ignore_index=True)
    print('Forecasts are complete.\n')
    return df_hist_plus_forecast