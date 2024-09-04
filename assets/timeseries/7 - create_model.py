import tensorflow as tf, keras, os
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.layers.merge import concatenate

def create_model(df_cnt, df_cat, df_y, cnt_var, config, inputs, embeds, test_days=0, future_days=1, include_cat=True):
    """
    Create various model definitions, fit each one to the data, and select the best model based on lowest loss.

    - df_cnt = continuous variables dataframe
    - df_cat= categorical variables dataframe 
    - df_y = dependent variable dataframe 
    - cnt_var = list of continuous variable names
    - config = model configurations
    - inputs = input specifications for modeling
    - embeds =  embeddings of categorical data for modeling
    - test_days = number of test days; default = 0
    - future_days = number of future days to predict; default = 1
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _, lstm_nodes, n_epochs, n_batch, n_levels, n_drop, _ = config
    input_cnt = Input(shape=(df_cnt.shape[1], df_cnt.shape[2]))
    if include_cat == True:
        input_all = concatenate([embeds[0], embeds[1], embeds[2],
         embeds[3], embeds[4], embeds[5],
         input_cnt])
    else:
        input_all = input_cnt
    visible_all = LSTM((int(lstm_nodes)), activation='relu', return_sequences=True)(input_all)
    hidden1_all = LSTM((int(lstm_nodes / 2)), activation='relu', return_sequences=True)(visible_all)
    hidden2_all = LSTM((int(lstm_nodes / 4)), activation='relu', return_sequences=True)(hidden1_all)
    hidden3_all = LSTM((int(lstm_nodes / 8)), activation='relu', return_sequences=True)(hidden2_all)
    if n_levels == 4:
        hidden4_all = LSTM((int(lstm_nodes / 16)), activation='relu')(hidden3_all)
        dropout_all = Dropout(n_drop)(hidden4_all)
    else:
        if n_levels == 5:
            hidden4_all = LSTM((int(lstm_nodes / 16)), activation='relu', return_sequences=True)(hidden3_all)
            hidden5_all = LSTM((int(lstm_nodes / 32)), activation='relu')(hidden4_all)
            dropout_all = Dropout(n_drop)(hidden5_all)
        else:
            if n_levels == 6:
                hidden4_all = LSTM((int(lstm_nodes / 16)), activation='relu', return_sequences=True)(hidden3_all)
                hidden5_all = LSTM((int(lstm_nodes / 8)), activation='relu', return_sequences=True)(hidden4_all)
                hidden6_all = LSTM((int(lstm_nodes / 4)), activation='relu')(hidden5_all)
                dropout_all = Dropout(n_drop)(hidden6_all)
            else:
                hidden4_all = LSTM((int(lstm_nodes / 16)), activation='relu')(hidden3_all)
                dropout_all = Dropout(n_drop)(hidden4_all)
    output_all = Dense(future_days)(dropout_all)
    if include_cat == True:
        model_all = Model(inputs=[inputs[0], inputs[1], inputs[2],
         inputs[3], inputs[4], inputs[5],
         input_cnt],
          outputs=[output_all])
    else:
        model_all = Model(inputs=[input_cnt], outputs=[output_all])
    model_all.compile(loss='mean_squared_error', optimizer='adam')
    if include_cat == True:
        model_all.fit([df_cat[0], df_cat[1], df_cat[2],
         df_cat[3], df_cat[4], df_cat[5],
         df_cnt],
          df_y, epochs=(int(n_epochs)),
          batch_size=(int(n_batch)),
          callbacks=[
         EarlyStopping(monitor='loss', patience=10)],
          verbose=0,
          shuffle=False)
    else:
        model_all.fit(df_cnt, df_y, epochs=(int(n_epochs)),
          batch_size=(int(n_batch)),
          callbacks=[
         EarlyStopping(monitor='loss', patience=10)],
          verbose=0,
          shuffle=False)
    return model_all