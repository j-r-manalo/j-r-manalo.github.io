import numpy as np, pandas as pd, re

def clean_text(content_unprocessed):
    """
    Cleans text from spaces and special characters
    """
    cleantext = content_unprocessed
    cleantext = re.sub('[^\\w\\s]', '', cleantext, re.UNICODE)
    cleantext = cleantext.lower()
    cleantext = cleantext.replace(' ', '')
    return cleantext


def fix_device(df):
    """
    Cleans messy device field
    """
    df['Device'] = df['Device'].str.lower()
    df['Device2'] = np.where((df.Device.str.contains('unknown', regex=False) == True) | (df.Device.str.contains('- n/a -', regex=False) == True) | (df.Device.str.contains('error', regex=False) == True), 'unknown', '')
    df['Device2'] = np.where(((df.Device.str.contains('phone', regex=False) == True) | (df.Device.str.contains('mobile', regex=False) == True) | (df.Device.str.contains('blackberry', regex=False) == True)) & (df.Device != '(not mobile)'), 'mobile', df['Device2'])
    df['Device2'] = np.where(((df.Device.str.contains('tablet', regex=False) == True) | (df.Device.str.contains('ipad', regex=False) == True)) & (df.Device != 'mobile/tablet'), 'tablet', df['Device2'])
    df['Device2'] = np.where((df.Device.str.contains('desktop', regex=False) == True) | (df.Device.str.contains('pc', regex=False) == True) | (df.Device.str.contains('personal computer', regex=False) == True), 'desktop', df['Device2'])
    df['Device2'] = np.where(df['Device2'] == '', 'other', df['Device2'])
    return df


def fix_partner(df):
    """
    Cleans messy partner field
    """
    # excluded for confidentiality
    return df