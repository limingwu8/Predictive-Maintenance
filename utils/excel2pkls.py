import pandas as pd
import numpy as np
import re
import pickle
import os
from datetime import datetime

def excel2pkl(path, sheetname, column_keep, sensor_names, save_path):
    '''
    Read data from excel and preprocess it, then save them to pickle for future reading
    :param path: path of a single excel file
    :param sheetname: specify which sheet should read, e.g. 'Oct17'
    :param column_keep: specify which column to keep, for some state sensor, just throw them
    :param sensor_names: sensor name of the remaining columns
    :param data_name: save processed data to pickle, names as data_name
    :return:.....
    '''
    excel = pd.ExcelFile(path)
    for sheet in range(len(sheetname)):

        df = pd.read_excel(excel, sheet_name = sheetname[sheet])
        # detele some columns
        df = df.iloc[:, column_keep]
        # find which column is for time, which column is for value
        time_index = np.arange(df.shape[1])[::2]
        value_index = np.arange(df.shape[1])[1::2]
        # read time column and value column
        time_col = df.iloc[:, time_index]
        value_col = df.iloc[:, value_index]
        # get number of values for each sensor, actually not necessary since we can remove NAs instead
        num_values = value_col.iloc[0, :]
        # clean the first row which are not values
        time_col = time_col.drop(0)
        value_col = value_col.drop(0)
        # drop the NA data and 'Bad' data, then add each time value pair to a list
        dfs = {}
        for i in range(len(sensor_names)):
            temp = pd.DataFrame({'time':time_col.iloc[:,i], 'value':value_col.iloc[:,i]})
            temp = temp[~temp['value'].isin(['Bad','I/O Timeout', '', ' ', 'Resize to show all values'])]
            temp = temp.dropna()
            temp = temp.reset_index(drop = True)
            timestr = temp['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            # convert time string to time
            temp['time'] = pd.to_datetime(timestr)
            # convert value string to numeric
            temp['value'] = pd.to_numeric(temp['value'])
            # remove duplicated rows if their time are the same
            temp.drop_duplicates(subset='time',inplace=True)

            dfs[sensor_names[i]] = temp

        # save the data to pickle
        save_file_name = os.path.join(save_path, sheetname[sheet] + '.pkl')
        with open(save_file_name,'wb') as f:
            pickle.dump(dfs, f)

if __name__ == '__main__':

    column_keep = [25, 26, 27, 28, 29, 30, 43, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
    sensor_names = ['MAIN_FILTER_IN_PRESSURE', 'MAIN_FILTER_OIL_TEMP', 'MAIN_FILTER_OUT_PRESSURE',
                    'OIL_RETURN_TEMPERATURE',
                    'TANK_FILTER_IN_PRESSURE', 'TANK_FILTER_OUT_PRESSURE', 'TANK_LEVEL', 'TANK_TEMPERATURE', 'FT-202B',
                    'FT-204B', 'PT-203', 'PT-204']

    # process file SepOctNov HSM Furnace tags.xlsx
    path = 'Y:\\USS-RF-Fan-Data-Analytics\\_8_Data-from-collaborator\\October 2017\\SepOctNov HSM Furnace tags.xlsx'
    sheetname = ['Sep17','Oct17', 'Nov17']
    save_path = '..\\dataset\\pkls'
    excel2pkl(path, sheetname, column_keep, sensor_names, save_path)

    # process file FCE_DATA_17SEP_MAR_With_Tag_Attributes
    path = 'Y:\\USS-RF-Fan-Data-Analytics\\_8_Data-from-collaborator\\October 2017\\FCE_DATA_17SEP_MAR_With_Tag_Attributes.xlsx'
    sheetname = ['1709']
    save_path = '..\\dataset\\pkls'
    excel2pkl(path, sheetname, column_keep, sensor_names, save_path)

    # FCE_DATA_17JUN_AUG
    path = 'Y:\\USS-RF-Fan-Data-Analytics\\_8_Data-from-collaborator\\October 2017\\FCE_DATA_17JUN_AUG.xlsx'
    sheetname = ['1706','1708']
    save_path = '..\\dataset\\pkls'
    excel2pkl(path, sheetname, column_keep, sensor_names, save_path)

    # FCE_DATA_17APR_MAY
    path = 'Y:\\USS-RF-Fan-Data-Analytics\\_8_Data-from-collaborator\\October 2017\\FCE_DATA_17APR_MAY.xlsx'
    sheetname = ['1705']
    save_path = '..\\dataset\\pkls'
    excel2pkl(path, sheetname, column_keep, sensor_names, save_path)

    path = 'Y:\\USS-RF-Fan-Data-Analytics\\_8_Data-from-collaborator\\Static Copy of HSM Furnace tags.xlsx'
    sheetname = ['HSM Furnace 2 tags']
    save_path = '..\\dataset\\pkls'
    save_name = '1707'
    excel2pkl(path, sheetname, column_keep, sensor_names, save_path)

