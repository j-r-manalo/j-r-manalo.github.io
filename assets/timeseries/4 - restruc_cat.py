import numpy as np, pandas as pd, preproc as pp

def restruc_cat(df, config, test_days=0, future_days=1):
    """
    Pre-process the data and restructure the categorical variables, so it can be read by tensorflow.

    - test_days = number of test days; default = 0
    - future_days = number of future days to predict; default = 1
    - n_days = number of prior days to include in the model predictors
    """
    n_days, _, _, _, _, _, _ = config
    categorical_vars = [
     'year', 'quarter', 'month',
     'weekday', 'weekend_flag', 'holiday_flag']
    df = df[categorical_vars]
    trn_X_cat, tst_X_cat = pp.refrac_cat(df, test_days=test_days, categorical_vars=categorical_vars)
    inputs = []
    embeddings = []
    for categorical_var in categorical_vars:
        input_spec, emb = pp.categorical_emb(df, categorical_var)
        inputs.append(input_spec)
        embeddings.append(emb)

    print('Restructuring the categorical data...\n')
    trn_X_cat_lst = pp.X_cat_lst(trn_X_cat, future_days + int(n_days) - 1)
    tst_X_cat_lst = pp.X_cat_lst(tst_X_cat, future_days + int(n_days) - 1)
    print('FYI:')
    print('\t- After restructuring, the categorical data shape is ', trn_X_cat_lst[1].shape)
    print('\nThe categorical data is now ready for modeling...\n')
    return (
     inputs, embeddings, trn_X_cat_lst, tst_X_cat_lst)