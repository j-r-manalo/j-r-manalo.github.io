import numpy as np, pandas as pd, clean_data as cd, preproc as pp, holiday_calendar as hc

def create_vars(df):
    """ 
    Creates variables used for the models and outputs summarized data
    """
    ivar = list(df.columns.drop(['Date', 'Impressions', 'Revenue']))
    print('Data is being cleaned...\n')
    for var in ivar:
        if var == 'Device':
            df = cd.fix_device(df)

    vars_long = []
    ivar = list(df.columns.drop([
     'Date', 'Impressions', 'Revenue', 'Device', 'Partner Unmapped', 'Partner']))
    for var in ivar:
        df[var] = df[var].apply(cd.clean_text)
        vars_cat = []
        vars_cat = df[var].unique()
        for var_cat in vars_cat:
            new_ivar = var + '_' + var_cat
            df[new_ivar] = np.where(df[var] == var_cat, df['Impressions'], 0)
            vars_long.append(new_ivar)

    df = df.sort_values(by=['Date']).reset_index(drop=True)
    df['year'] = df['Date'].apply(lambda x: x.year)
    df['quarter'] = df['Date'].apply(lambda x: x.quarter)
    df['month'] = df['Date'].apply(lambda x: x.month)
    df['day'] = df['Date'].apply(lambda x: x.day)
    df['weekday'] = df['Date'].apply(lambda x: x.dayofweek)
    df['weekend_flag'] = np.where(df['weekday'] >= 5, 1, 0)
    hol_cal = hc.holiday_calendar(df)
    df = pd.merge(df, hol_cal,
      left_on='Date',
      right_on='Date',
      how='left')
    df['Holiday'].fillna('Not Holiday', inplace=True)
    df['holiday_flag'] = np.where(df['Holiday'] != 'Not Holiday', 1, 0)
    print('Independent variables in their raw form have been created and summarized...\n')
    df2 = df.groupby(by=['year', 'quarter', 'month', 'Partner_Mapped', 'Publisher', 'Device'], as_index=False)[['Impressions', 'Revenue']].apply(lambda x: x.sum())
    df2.sort_values(by=['year', 'quarter', 'month'], inplace=True)
    df2 = df2.reset_index()
    df = df.groupby(by=['Date', 'year', 'quarter', 'month', 'weekday', 'weekend_flag', 'holiday_flag'], as_index=False)[(['Impressions', 'Revenue'] + vars_long)].apply(lambda x: x.sum())
    df.sort_values(by=['Date'], inplace=True)
    df = df.reset_index()
    print('The data is now ready for pre-processing and restructuring...\n')
    return (
     df, df2, vars_long)