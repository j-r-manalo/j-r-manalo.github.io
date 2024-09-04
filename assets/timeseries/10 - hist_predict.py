import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta, datetime

def hist_predict(df_cnt, df_cat, model, trn_dates, y_scaler):
    """
    Pre-process the data and restructure the continuous variables, so it can be read by tensorflow.

    - df_cnt = continuous variables dataframe
    - df_cat = categorical variables dataframe
    - model = best model
    - trn_dates = dates the model was trained on
    - y_scaler = scalers for dependent variable (Revenue); used for normalization after modeling 
    """
    pd.options.mode.chained_assignment = None
    trn_X_p = model.predict([df_cat[0], df_cat[1], df_cat[2],
     df_cat[3], df_cat[4], df_cat[5],
     df_cnt])
    trn_X_p_fut = trn_X_p[(-1, )]
    trn_X_p_hist = trn_X_p[:-1, 0]
    trn_X_p_all = np.append(trn_X_p_hist, trn_X_p_fut)
    trn_X_p_all = trn_X_p_all.reshape(-1, 1)
    trn_X_p_rescale = y_scaler.inverse_transform(trn_X_p_all)
    df_hist_predict = pd.DataFrame(trn_X_p_rescale)
    df_hist_predict.columns = ['Revenue Prediction']
    df_hist_predict['Revenue Prediction'] = df_hist_predict['Revenue Prediction'].astype(int)
    trn_dates2 = pd.DataFrame(trn_dates)
    lastindex_fit = df_hist_predict.shape[0] - 1
    lastindex_full = trn_dates2.shape[0] - 1
    index_shift = lastindex_full - lastindex_fit
    for i in range(0, lastindex_fit + 1):
        df_hist_predict.loc[(i, 'Date')] = trn_dates2.loc[(i, 'Date')] + timedelta(index_shift + 1)

    return df_hist_predict