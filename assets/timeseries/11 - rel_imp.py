import keras, tensorflow as tf, os, pandas as pd, sys, numpy as np, shap
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def rel_imp(df_cnt, trn_dates, model, cnt_var, config):
    """
    Find Relative Importance across input variables

    - df_cnt = continuous variables dataframe
    - trn_dates = dates the model was trained on
    - model = model without embeddings
    - cnt_var = list of continuous variable names
    - config = model configurations
    """
    n_days, _, _, _, _, _, _ = config
    explainer = shap.DeepExplainer(model, df_cnt)
    shapval = np.array(explainer.shap_values(df_cnt))
    print(shapval.shape)
    shapval2 = np.reshape(shapval, (shapval.shape[1], shapval.shape[3]))
    print(shapval2)
    df_shap = pd.DataFrame(shapval2)
    print(df_shap.head())
    df_shap2 = df_shap.iloc[:, int(n_days):df_shap.shape[1]]
    df_shap2.columns = cnt_var
    print(df_shap.head())
    trn_dates = pd.DataFrame(trn_dates)
    print(trn_dates.head())
    print(trn_dates.tail())
    df_shap2['Date'] = trn_dates
    print(df_shap2.head())
    print(df_shap2.tail())
    df_shap2['year'] = df_shap2['Date'].apply(lambda x: x.year)
    df_shap2['quarter'] = df_shap2['Date'].apply(lambda x: x.quarter)
    df_shap2['month'] = df_shap2['Date'].apply(lambda x: x.month)
    df_shap2 = df_shap2[(df_shap2['year'] >= df_shap2['year'].max() - 1)]
    df_shap3 = pd.DataFrame(df_shap2.groupby(by=['year', 'quarter', 'month'], as_index=False)[cnt_var].apply(lambda x: abs(x).mean()).stack())
    df_shap3 = df_shap3.reset_index()
    print(df_shap3.head())
    df_shap3['category'] = df_shap3['level_3'].apply(lambda x: x.rsplit('_', 1)[0]).str.replace('Device2', 'Device')
    # other variables excluded for confidentiality
    print(df_shap3.head())
    df_shap3['unit'] = df_shap3['level_3'].apply(lambda x: x.rsplit('_', 1)[1]).str.upper()
    df_shap3 = df_shap3.drop(['level_3'], axis=1)
    print('Relative Importance calculations are complete...\n')
    return df_shap3