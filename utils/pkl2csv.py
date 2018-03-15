import os
import pickle
import errno
from concatenate_and_resample import resample,concatenate,check_duplicate_index

# combine all pickles files and write to CSV, each sensor save as one CSV
def pkl2csv(pkl_path, csv_root, interval):

    pkls_path = [os.path.join(pkl_path, i) for i in os.listdir(pkl_path)]

    concatenated = []
    for path in pkls_path:
        with open(path,'rb') as f:
            dfs = pickle.load(f)
            concatenated = concatenate(concatenated, dfs)

    # resample each element in concatenated to a specific interval
    for key in concatenated.keys():
        # convert index to a specific format ('%Y-%m-%d %H-%M-%S')
        temp = resample(concatenated[key], interval = interval)
        time = temp.index.strftime('%Y-%m-%d %H:%M:%S')
        temp.index = time
        concatenated[key] = temp


    if interval == 'D':
        folder_name = 'sample_1_day'
    elif interval == 'H':
        folder_name = 'sample_1_hour'
    elif interval == '6H':
        folder_name = 'sample_6_hour'
    elif interval =='12H':
        folder_name = 'sample_12_hour'
    elif interval == '18H':
        folder_name = 'sample_18_hour'
    else:
        print('please input correct interval')
        return

    for key in dfs:
        csv_path = os.path.join(csv_root,'sampled',folder_name, key + '.csv')
        if not os.path.exists(os.path.dirname(csv_path)):
            try:
                os.makedirs(os.path.dirname(csv_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise exc
        concatenated[key].to_csv(csv_path, sep=',', encoding='utf-8', index_label='time')



if __name__ == '__main__':
    # pkl_path = '..\\dataset\\pkls\\1705.pkl'
    # csv_path = '..\\dataset\\csv\\'
    # pkl2csv(pkl_path, csv_path)

    # pkl_path = '..\\dataset\\pkls\\1706.pkl'
    # csv_path = '..\\dataset\\csv\\'
    # pkl2csv(pkl_path, csv_path)
    #
    # pkl_path = '..\\dataset\\pkls\\1707.pkl'
    # csv_path = '..\\dataset\\csv\\'
    # pkl2csv(pkl_path, csv_path)
    #
    # pkl_path = '..\\dataset\\pkls\\1708.pkl'
    # csv_path = '..\\dataset\\csv\\'
    # pkl2csv(pkl_path, csv_path)
    #
    # pkl_path = '..\\dataset\\pkls\\1709.pkl'
    # csv_path = '..\\dataset\\csv\\'
    # pkl2csv(pkl_path, csv_path)
    #
    # pkl_path = '..\\dataset\\pkls\\1710.pkl'
    # csv_path = '..\\dataset\\csv\\'
    # pkl2csv(pkl_path, csv_path)
    #
    # pkl_path = '..\\dataset\\pkls\\1711.pkl'
    # csv_path = '..\\dataset\\csv\\'
    # pkl2csv(pkl_path, csv_path)

    pkl_path = '..\\dataset\\pkls\\'
    csv_root = '..\\dataset\\csv\\'
    interval = 'D'
    pkl2csv(pkl_path, csv_root, interval)