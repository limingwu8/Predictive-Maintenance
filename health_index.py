from pandas import Series
from pandas import read_csv
from pandas import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import os
import numpy as np

save_info = 1
# Root path to save the generated figure
root_path = '/home/liming/Dropbox/US_Steel/USS-RF-Fan-Data-Analytics/_13_Preliminary-results/LSTM-preciction/multi-step-prediction/compare-v3/health_index'
sensor_names = {
    'MAIN_FILTER_IN_PRESSURE','MAIN_FILTER_OIL_TEMP','MAIN_FILTER_OUT_PRESSURE','OIL_RETURN_TEMPERATURE',
    'TANK_FILTER_IN_PRESSURE','TANK_FILTER_OUT_PRESSURE','TANK_LEVEL','TANK_TEMPERATURE','FT-202B',
    'FT-204B','PT-203','PT-204'
}
operating_range = {
    'MAIN_FILTER_IN_PRESSURE':(65,40,90),'MAIN_FILTER_OIL_TEMP':(110,90,130),'MAIN_FILTER_OUT_PRESSURE':(63,40,90),
    'OIL_RETURN_TEMPERATURE':(110,90,130),'TANK_FILTER_IN_PRESSURE':(20,10,30),'TANK_FILTER_OUT_PRESSURE':(18,10,30),
    'TANK_LEVEL':(19,14,23),'TANK_TEMPERATURE':(110,90,130),'FT-202B':(0.5,0,3.5),'FT-204B':(0.5,0,3.5),
    'PT-203':(0.5,0,3.5),'PT-204':(0.5,0,3.5)
}
weights = {
    'MAIN_FILTER_IN_PRESSURE':1,'MAIN_FILTER_OIL_TEMP':1,'MAIN_FILTER_OUT_PRESSURE':0,
    'OIL_RETURN_TEMPERATURE':1,'TANK_FILTER_IN_PRESSURE':0,'TANK_FILTER_OUT_PRESSURE':0,
    'TANK_LEVEL':1,'TANK_TEMPERATURE':1,'FT-202B':1,'FT-204B':1,
    'PT-203':1,'PT-204':1
}
line_colors = {
    'MAIN_FILTER_IN_PRESSURE':'b','MAIN_FILTER_OIL_TEMP':'g','MAIN_FILTER_OUT_PRESSURE':'r',
    'OIL_RETURN_TEMPERATURE':'c','TANK_FILTER_IN_PRESSURE':'m','TANK_FILTER_OUT_PRESSURE':'y',
    'TANK_LEVEL':'k','TANK_TEMPERATURE':'brown','FT-202B':'pink','FT-204B':'gray',
    'PT-203':'orange','PT-204':'purple'
}
sensor_acronym = {
    'MAIN_FILTER_IN_PRESSURE': 'P1', 'MAIN_FILTER_OIL_TEMP': 'T1', 'MAIN_FILTER_OUT_PRESSURE': 'P2',
    'OIL_RETURN_TEMPERATURE': 'T2', 'TANK_FILTER_IN_PRESSURE': 'P3', 'TANK_FILTER_OUT_PRESSURE': 'P4',
    'TANK_LEVEL': 'L1', 'TANK_TEMPERATURE': 'T3', 'FT-202B': 'V1', 'FT-204B': 'V2', 'PT-203': 'V3', 'PT-204': 'V4'
}

def load_dataset(paths):
    series = {}
    sensor_names = []
    for path in paths:
        serie = read_csv(path, sep=',')

        serie.time = [datetime.datetime.strptime(
            i, "%Y-%m-%d") for i in serie.time]
        sensor_name = os.path.basename(path).split('.')[0]
        series[sensor_name] = serie
    return series

def get_paths():
    root_health_index_all = os.path.join(os.curdir, 'health_index', 'health_index_all')
    files = os.listdir(root_health_index_all)
    paths_health_index_all = [os.path.join(root_health_index_all, s) for s in files]

    root_health_index_pred = os.path.join(os.curdir, 'health_index', 'health_index_pred')
    files = os.listdir(root_health_index_pred)
    paths_health_index_pred = [os.path.join(root_health_index_pred, s) for s in files]

    return paths_health_index_all, paths_health_index_pred

def plot_health_index_combined(series, overall_health_index, path):
    """
    Combine all health index in one figure
    """
    label_fontsize = 35
    legend_fontsize = 18
    axis_fontsize = 30
    linewidth = 3

    fig = plt.figure()
    axis = fig.add_subplot(1,1,1)
    # axis.xaxis_date()
    # axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    for key in series.keys():
        if key in ['MAIN_FILTER_OUT_PRESSURE','TANK_FILTER_IN_PRESSURE','TANK_FILTER_OUT_PRESSURE']:
            continue
        # plt.plot(series[key].time,series[key].health_index, label=sensor_acronym[key], linewidth=linewidth,alpha=0.3, color=line_colors[key])
        plt.plot(np.arange(len(series[key].health_index)), series[key].health_index, label=sensor_acronym[key], linewidth=linewidth,alpha=0.3, color=line_colors[key])
    plt.plot(np.arange(len(overall_health_index.values)),overall_health_index.values, label = 'overall', linewidth=linewidth+3, color = 'black')
    plt.xlabel('Days', fontsize=label_fontsize)
    plt.ylabel('Health Index', fontsize=label_fontsize)
    plt.title('Health Index', fontsize=label_fontsize)
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.legend(fontsize=legend_fontsize,bbox_to_anchor=(1.13,0.5), loc="center right")
    # plt.show()
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    if save_info:
        fig.savefig(os.path.join(path, 'health_index_combined.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_health_index_separated(series, path):
    """
    For each health index, generate a figure
    """
    label_fontsize = 35
    legend_fontsize = 20
    axis_fontsize = 30
    linewidth = 5

    for key in series.keys():
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.xaxis_date()
        axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        plt.plot(series[key].time,series[key].health_index, label=sensor_acronym[key], linewidth=linewidth)
        # plt.plot(overall_health_index, label='overall_health_index', linewidth=linewidth)
        plt.xlabel('Health Index', fontsize=label_fontsize)
        plt.ylabel('Health Index', fontsize=label_fontsize)
        axis.set_ylim([0, 1])
        plt.title('Health Index: ' + sensor_acronym[key], fontsize=label_fontsize)
        plt.xticks(fontsize=axis_fontsize)
        plt.yticks(fontsize=axis_fontsize)
        plt.legend(fontsize=legend_fontsize)
        # plt.show()
        if save_info:
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(os.path.join(path, 'health_index_' + key + '.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)
    # plot overall health index
    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1)
    # axis.xaxis_date()
    # axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    # plt.plot(overall_health_index, label='overall_health_index', linewidth=linewidth)
    # plt.xlabel('Days', fontsize=label_fontsize)
    # plt.ylabel('Health Index', fontsize=label_fontsize)
    # plt.title('Overall Health Index', fontsize=label_fontsize)
    # plt.xticks(fontsize=axis_fontsize)
    # plt.yticks(fontsize=axis_fontsize)
    # plt.show()
    # fig.set_size_inches(18.5, 10.5)
    # fig.savefig(os.path.join(root_path, 'health_index_overall.png'), bbox_inches='tight', dpi=150)
    # plt.close(fig)

def plot_health_index_overlay(series_all,series_pred, path):
    label_fontsize = 35
    legend_fontsize = 20
    axis_fontsize = 30
    linewidth = 5

    for key in series_all.keys():
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.xaxis_date()
        axis.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        plt.plot(series_all[key].time,series_all[key].health_index, label='ground truth health index', linewidth=linewidth)
        plt.plot(series_pred[key].time,series_pred[key].health_index, label='predicted health index', linewidth=linewidth)
        # plt.plot(overall_health_index, label='overall_health_index', linewidth=linewidth)
        plt.xlabel('Health Index', fontsize=label_fontsize)
        plt.ylabel('Health Index', fontsize=label_fontsize)
        axis.set_ylim([0,1])
        plt.title('Health Index: ' + key, fontsize=label_fontsize)
        plt.xticks(fontsize=axis_fontsize)
        plt.yticks(fontsize=axis_fontsize)
        plt.legend(fontsize=legend_fontsize)
        # plt.show()
        if save_info:
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(os.path.join(path, 'health_index_' + key + '.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)


def get_combined_health_index(series):
    """
    plot all health index in one figure
    """
    # health_indices = [series[key].health_index_pred.values for key in series.keys()]
    #
    # for i in zip(health_indices[0], health_indices[1],health_indices[2],
    #              health_indices[3],health_indices[4],health_indices[5],
    #              health_indices[6],health_indices[7],health_indices[8],
    #              health_indices[9],health_indices[10],health_indices[11]):
    #     print(i)
    df = pd.DataFrame()
    index = None
    for key in series.keys():
        df = pd.concat([df,pd.DataFrame({key:series[key].health_index})], axis=1)
    # df = pd.concat([df,pd.DataFrame({'aaa':[2,3,4,5]})],axis=1)

    overall_health_index = []
    for i, row in df.iterrows():
        data = Series.to_dict(row)
        s = 0
        weights2 = {}
        for key in data.keys():
            weights2[key] = weights[key]*abs(0.5-data[key])
            s = s + data[key]*weights2[key]
        s = s/sum(weights2.values())
        # s = s/sum(data*)
        overall_health_index.append(s)
    overall_health_index = pd.DataFrame({'overall_health_index':overall_health_index},index = series['PT-204'].time)

    return overall_health_index

if __name__ == '__main__':
    plot = 'pred'
    paths_health_index_all,paths_health_index_pred = get_paths()
    series_all = load_dataset(paths_health_index_all)
    series_pred = load_dataset(paths_health_index_pred)
    if plot == 'pred':
        path = os.path.join(root_path,'health_index_pred')
        overall_health_index = get_combined_health_index(series_pred)
        plot_health_index_combined(series_pred, overall_health_index, path)
        plot_health_index_separated(series_pred, path)
    elif plot=='all':
        path = os.path.join(root_path,'health_index_all')
        overall_health_index = get_combined_health_index(series_all)
        plot_health_index_combined(series_all, overall_health_index, path)
        plot_health_index_separated(series_all, path)
    elif plot == 'overlay':
        path = os.path.join(root_path,'health_index_overlay')
        plot_health_index_overlay(series_all,series_pred, path)
