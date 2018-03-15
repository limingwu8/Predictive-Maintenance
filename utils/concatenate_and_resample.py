import pandas as pd
import numpy as np
'''
resample time series data
Alias	Description
D	calendar day frequency
W	weekly frequency
M	month end frequency
SM	semi-month end frequency (15th and end of month)
MS	month start frequency
Q	quarter end frequency
H	hourly frequency
T, min	minutely frequency
S	secondly frequency
http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
'''
def resample(df, interval = 'D'):
    # set time column as index
    df.set_index('time',inplace = True)
    # drop duplicated time
    keep_index = ~df.index.duplicated()
    df = df.iloc[keep_index, :]
    idx = pd.date_range(start = df.index[0],end = df.index[df.index.__len__()-1], freq = 'S')
    # reindex to every second
    df = df.reindex(idx, fill_value = None)
    df.fillna(method = 'ffill', inplace = True)
    # resample to every interval
    df = df.resample(interval).pad()
    df.fillna(method = 'backfill', inplace = True)

    return df

def concatenate(dfs1, dfs2):
    if dfs1.__len__() == 0:
        return dfs2
    else:
        dfs = {}
        for key in dfs1.keys():
            dfs[key] = pd.concat([dfs1[key],dfs2[key]])
        return dfs
'''
Test function, check if there are duplicated index in dfs
dfs is a dictionary, each element is a dataframe
'''
def check_duplicate_index(dfs):
    duplicated = []
    for key in dfs.keys():
        df = dfs[key]
        duplicated.append(np.where(df.index.duplicated()==True))
    return duplicated
if __name__ == '__main__':
    resample()