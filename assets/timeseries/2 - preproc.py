from sklearn.preprocessing import MinMaxScaler
import numpy as np, pandas as pd
from keras.layers import Input, Embedding

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Reframes dataset for time series
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += ['var%d(t-%d)' % (j + 1, i) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += ['var%d(t)' % (j + 1) for j in range(n_vars)]
        else:
            names += ['var%d(t+%d)' % (j + 1, i) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def refrac_cat(processed_data, test_days, categorical_vars):
    """
    Preproceess categorical variables
    """
    trn_size = len(processed_data) - test_days
    print('Refactoring the categorical variables and splitting the training and testing data...\n')
    input_list_trn = []
    input_list_tst = []
    for c in categorical_vars:
        jjj = np.asarray(processed_data[c].tolist())
        jjj_trn = pd.factorize(jjj[0:trn_size])[0]
        jjj_tst = pd.factorize(jjj[trn_size:len(processed_data)])[0]
        input_list_trn.append(np.asarray(jjj_trn))
        input_list_tst.append(np.asarray(jjj_tst))

    return (input_list_trn, input_list_tst)


def categorical_emb(df, categorical_var):
    """
    Create inputs and embeddings for modeling
    """
    input_cat = Input(shape=(1, ))
    no_of_unique_cat = df[categorical_var].nunique()
    embedding_size = min(np.ceil(no_of_unique_cat / 2), 50)
    embedding_size = int(embedding_size)
    vocab = no_of_unique_cat + 1
    emb = Embedding(vocab, embedding_size, input_length=1)(input_cat)
    return (input_cat, emb)


def X_cat_lst(in_lst, future_days):
    """
    Reformat categorical data
    """
    out_lst = []
    for cat_var in in_lst:
        outdata = pd.DataFrame(cat_var)
        outdata = outdata.values
        if future_days == 0:
            outdata = outdata
        else:
            outdata = outdata[:-future_days]
        out_lst.append(outdata)

    return out_lst