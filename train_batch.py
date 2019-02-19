# from RNN.time_series_prediction.Sensor import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from Sensor import Sensors

import os
# configuration
save_info = 1       # 1: save information in file, 0: do not save
train = 0           # 1: train model, 0: load model

n_lag = 1
n_epochs = 100
dataset_path = os.path.join(os.curdir, 'dataset','csv', 'sampled')
root_path = '/home/PNW/wu1114/Dropbox/US_Steel/USS-RF-Fan-Data-Analytics/_13_Preliminary-results/LSTM-preciction/multi-step-prediction/compare-v3/'
sensor_names = {
    'MAIN_FILTER_IN_PRESSURE','MAIN_FILTER_OIL_TEMP','MAIN_FILTER_OUT_PRESSURE','OIL_RETURN_TEMPERATURE',
    'TANK_FILTER_IN_PRESSURE','TANK_FILTER_OUT_PRESSURE','TANK_LEVEL','TANK_TEMPERATURE','FT-202B',
    'FT-204B','PT-203','PT-204'
}
sample_rates_n_seq = {
    'sample_1_hour':(1,48), 'sample_6_hour':(1,8), 'sample_12_hour':(1,4),
    'sample_18_hour':(1,2), 'sample_1_day':(1,2)
}
for name in sensor_names:
    # if name!='MAIN_FILTER_OIL_TEMP':
    #     continue
    for j in sample_rates_n_seq:
        n_seqs = sample_rates_n_seq[j]
        sample_rate = j
        if j != 'sample_1_day':
            continue
        if name in ['MAIN_FILTER_OUT_PRESSURE', 'TANK_FILTER_IN_PRESSURE', 'TANK_FILTER_OUT_PRESSURE']:
            continue
        for s in n_seqs:
            if s != 1:
                continue
            S = Sensors(n_seq = s, n_epochs= n_epochs, dataset_path = dataset_path, sensor_name = name,
                       sample_rate = sample_rate, train = train, root_path = root_path, save_info = save_info)
            if train == 1:
                S.run_train()   # train the network
            else:
                S.load_model_and_predict()  # load .h5 file and make prediction
                # S._normalize()
                # S.get_pred_health_score()
