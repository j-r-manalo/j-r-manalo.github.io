import pandas as pd, rel_imp as ri, create_model as cm, hist_predict as hp, forecasts as fc, restruc_cat as rca, restruc_cnt as rcn, tensorflow as tf, keras, os
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.layers.merge import concatenate
from keras import backend as K

def model_configs():
    """
    Define the scope of the possible model confiturations

    n_days = number of prior days to include in the model lookback
    n_nodes = number of initial nodes
    n_epochs = number of epochs
    n_batch = number of batches 
    n_levels = number of the levels of the model. Only choices are 4, 5, and 6; 
                where 6 has 6 levels in a feedforward modeling structure
    n_drop = numbers for droprate  
    n_samp = number of samples to iterate the configuration over
    """
    n_days = [
     1, 3, 7]
    n_nodes = [128, 256, 512, 1024, 1536]
    n_epochs = [200]
    n_batch = [32]
    n_levels = [4, 5, 6]
    n_drop = [0.3, 0.4, 0.5]
    n_samp = list(range(1, 3))
    configs = list()
    for a in n_days:
        for b in n_nodes:
            for c in n_epochs:
                for d in n_batch:
                    for e in n_levels:
                        for f in n_drop:
                            for g in n_samp:
                                cfg = [
                                 a, b, c, d, e, f, g]
                                configs.append(cfg)

    print('The total number of configurations = %d' % len(configs))
    return configs


def score_model(df_cnt, df_cat, df_y, cnt_var, config, inputs, embeds):
    """
    Fit and score the various model configurations.

    - df_cnt = continuous variables dataframe
    - df_cat= categorical variables dataframe 
    - df_y = dependent variable dataframe 
    - cnt_var = list of continuous variable names
    - config = model configurations
    - inputs = input specifications for modeling
    - embeds =  embeddings of categorical data for modeling
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    n_days, n_nodes, n_epochs, n_batch, n_levels, n_drop, n_samp = config
    model = cm.create_model(df_cnt, df_cat, df_y, cnt_var, config, inputs=inputs, embeds=embeds)
    scores = model.evaluate([df_cat[0], df_cat[1], df_cat[2], df_cat[3],
     df_cat[4], df_cat[5], df_cnt],
      df_y, verbose=0)
    print(f"\nModel loss = {scores}\n")
    return (
     scores, n_days, n_nodes, n_epochs, n_batch, n_levels, n_drop, n_samp)


def grid_search(df, cnt_var, cfg_list):
    """
    Score each model across the different configurations of the model grid. 
    Save scores. Select best scoring model based on lowest loss.
    Score best model.
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('\nThe models are now being searched over for the best model...\n')
    score_results = pd.DataFrame(columns=[
     'n_days', 'n_nodes', 'n_epochs', 'n_batch', 'n_levels', 'n_drop', 'n_samp', 'loss'])
    for cfg in cfg_list:
        print(f"\nFitting model config = {cfg}")
        trn_X_cnt, trn_y, x_scaler, y_scaler, _, _, trn_dates, ivar, ivarr = rcn.restruc_cnt(df=df,
          config=cfg,
          cnt_var=cnt_var)
        inputs, embeds, trn_X_cat_lst, _ = rca.restruc_cat(df=df, config=cfg)
        loss, n_days, n_nodes, n_epochs, n_batch, n_levels, n_drop, n_samp = score_model(trn_X_cnt,
          trn_X_cat_lst, trn_y, cnt_var, cfg, inputs=inputs, embeds=embeds)
        score_results = score_results.append({'n_days':n_days, 
         'n_nodes':n_nodes,  'n_epochs':n_epochs,  'n_batch':n_batch,  'n_levels':n_levels,  'n_drop':n_drop, 
         'n_samp':n_samp,  'loss':loss},
          ignore_index=True)
        K.clear_session()

    score_results_agg = score_results.groupby(by=[
     'n_days', 'n_nodes', 'n_epochs', 'n_batch', 'n_levels', 'n_drop'],
      as_index=False).agg({'loss': 'mean'})
    score_results_agg['rank'] = score_results_agg['loss'].rank(method='max',
      na_option='bottom')
    score_results_agg.sort_values(by=[
     'rank', 'n_nodes', 'n_days'],
      inplace=True)
    score_results_agg.reset_index(drop=True, inplace=True)
    score_results_best = score_results_agg.iloc[0]
    best_config = [score_results_best['n_days'].astype(int), score_results_best['n_nodes'].astype(int),
     score_results_best['n_epochs'].astype(int), score_results_best['n_batch'].astype(int),
     score_results_best['n_levels'].astype(int), score_results_best['n_drop'], score_results_best['loss']]
    print('The best config is = ', best_config, '\n')
    if score_results_best['loss'] > 0.01:
        print('WARNING: The best config has an unacceptable loss of ', score_results_best['loss'], '.')
        print('WARNING: This is greater than our acceptable loss threshold of .01.')
        print('WARNING: You should consider manually fitting the model.\n')
    print('Running the model with the best config...\n')
    trn_X_cnt, trn_y, x_scaler, y_scaler, _, _, trn_dates, ivar, ivarr = rcn.restruc_cnt(df=df,
      config=best_config,
      cnt_var=cnt_var)
    inputs, embeds, trn_X_cat_lst, _ = rca.restruc_cat(df=df,
      config=best_config)
    best_model = cm.create_model(trn_X_cnt, trn_X_cat_lst, trn_y, cnt_var, best_config,
      inputs=inputs, embeds=embeds)
    df_hist_predict = hp.hist_predict(trn_X_cnt, trn_X_cat_lst, best_model, trn_dates, y_scaler)
    df_hist_plus_forecast = fc.forecasts(df, best_config, best_model, cnt_var, x_scaler,
      y_scaler, ivar, ivarr, test_days=0, future_days=1)
    df_hist_plus_forecast = pd.merge(df_hist_plus_forecast, df_hist_predict,
      left_on='Date',
      right_on='Date',
      how='left')
    print('Running Relative Importance calculations...\n')
    relimp_model = cm.create_model(trn_X_cnt, trn_X_cat_lst, trn_y, cnt_var, best_config,
      inputs=inputs, embeds=embeds, include_cat=False)
    rel_imp = ri.rel_imp(trn_X_cnt, trn_dates, relimp_model, cnt_var, best_config)
    K.clear_session()
    return (
     score_results_agg, best_model, df_hist_plus_forecast, rel_imp)