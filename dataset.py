import os
import wfdb
from scipy import signal
import numpy as np
import wfdb
import pandas as pd
from IPython.display import display
import random
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from collections import Counter
# from matplotlib import pyplot as plt
# from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset

from scipy.signal import resample
from sklearn.model_selection import train_test_split

from numpy.matlib import repmat

import sys

extract_labels = ['N', 'L', 'R', 'V', 'A', 'B', 's', 'T']

vals = [i for i in range(len(extract_labels))]
label_dict = dict(zip(extract_labels, vals))
print(label_dict)

vals2 = [0 for i in range(len(extract_labels))]
label_cnt_dict = dict(zip(extract_labels, vals2))
print(label_cnt_dict)

max_cnt = 5000

def normal_and_spit(normal_data_path, abnormal_data_path):
    normal_data = np.array(pd.read_csv(normal_data_path))
    abnormal_data = np.array(pd.read_csv(abnormal_data_path))

    print('before suffle', normal_data.shape)
    np.random.shuffle(normal_data)
    print('after shuffle', normal_data.shape)

    normal_num = len(normal_data)
    abnormal_num = len(abnormal_data)

    all_data = np.r_[normal_data, abnormal_data]
    min_val = np.min(all_data)
    max_val = np.max(all_data)

    normal_data = (normal_data - min_val) / (max_val - min_val)
    abnormal_data = (abnormal_data - min_val) / (max_val - min_val)

    normal_train = normal_data[:normal_num - abnormal_num, :]
    normal_test = normal_data[normal_num - abnormal_num:, :]

    pd.DataFrame(normal_train).to_csv('./data/normal_train.csv', index=False, header=False)
    pd.DataFrame(normal_test).to_csv('./data/normal_test.csv', index=False, header=False)
    pd.DataFrame(abnormal_data).to_csv('./data/abnormal_test.csv', index=False, header=False)


class AutoencoderDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.array(pd.read_csv(data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = torch.tensor(self.data[i], dtype=torch.float32).unsqueeze(0)  # require shape [1 256]
        return data

def extraction_mitbih(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = [i[:-4] for i in os.listdir(data_dir) if '.hea' in i]
    # print(filenames)

    # labels = []
    # for name in filenames:
    #     anno = wfdb.rdann(data_dir + '/' + name, 'atr')
    #     labels.extend((anno.symbol))
    # labels = Counter(labels)

    # for lab in extract_labels:
    #     if lab not in labels:
    #         print('Error! Not found label:', lab)

    # vals = [i for i in range(len(extract_labels))]
    # label_dict = dict(zip(extract_labels, vals))
    # print(label_dict)

    # import sys
    # sys.exit()

    savedata = []
    savelabels = []
    for name in filenames:
        # record = wfdb.rdheader(data_dir + '/' + name)
        # display(record.__dict__)
        # sys.exit()

        alldata = wfdb.rdrecord(data_dir + '/' + name).p_signal
        length, channels  = alldata.shape
        print('length', length, 'channels', channels)

        if max_idx > length:
            max_idx = length
        elif max_idx < radius * 2 + 1:
            max_idx = length
        anno = wfdb.rdann(data_dir + '/' + name, 'atr')
        peaks = anno.sample
        
        for peak in peaks:
            start = peak - radius
            end = peak + radius
            if start < 0 or end > max_idx:
                continue

            for ch in range(channels):
                if ch not in extract_channels:
                    continue

                tmpdata = wfdb.rdrecord(data_dir + '/' + name, sampfrom=start, sampto=end, channels=[ch]).p_signal
                tmpdata = np.array(tmpdata).squeeze()
                tmplabel = wfdb.rdann(data_dir + '/' + name, 'atr', sampfrom=start, sampto=end).symbol
                # print(tmpdata.shape, len(tmplabel))
                
                if len(tmplabel) == 1:
                    tmplabel = tmplabel[0]
                    
                    if tmplabel in extract_labels and label_cnt_dict[tmplabel] < max_cnt:
                        label_val = label_dict[tmplabel]

                        savedata.append(tmpdata)
                        savelabels.append(label_val)

                        label_cnt_dict[tmplabel] = label_cnt_dict[tmplabel] + 1
            

        print(name, len(savedata), len(savelabels), Counter(savelabels))
        # break
    savedata = np.array(savedata)
    savelabels = np.array(savelabels)
    print('finish all', savedata.shape, Counter(savelabels))
    pd.DataFrame(savedata).to_csv(save_dir + '/data_mitbih.csv', index=False, header=False)
    pd.DataFrame(savelabels).to_csv(save_dir + '/labels_mitbih.csv', index=False, header=False)


def extraction_holter(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = [i[:-4] for i in os.listdir(data_dir) if '.atr' in i]
    # print(filenames)

    # labels = []
    # for name in filenames:
    #     anno = wfdb.rdann(data_dir + '/' + name, 'atr')
    #     labels.extend((anno.symbol))
    # labels = Counter(labels)
    # print('all labels:', labels)

    # for lab in extract_labels:
    #     if lab not in labels:
    #         print('Error! Not found label:', lab)

    # vals = [i for i in range(len(extract_labels))]
    # label_dict = dict(zip(extract_labels, vals))
    # print(label_dict)

    # sys.exit()

    savedata = []
    savelabels = []
    for name in filenames:
        # record = wfdb.rdheader(data_dir + '/' + name)
        # display(record.__dict__)
        # sys.exit()

        alldata = wfdb.rdrecord(data_dir + '/' + name).p_signal
        length, channels  = alldata.shape
        print('length', length, 'channels', channels)

        if max_idx > length:
            max_idx = length
        elif max_idx < radius * 2 + 1:
            max_idx = length
        anno = wfdb.rdann(data_dir + '/' + name, 'atr')
        peaks = anno.sample
        # print(peaks)
        # sys.exit()
        
        for peak in peaks:
            start = peak - radius
            end = peak + radius
            if start < 0 or end > max_idx:
                continue

            for ch in range(channels):
                if ch not in extract_channels:
                    continue

                tmpdata = wfdb.rdrecord(data_dir + '/' + name, sampfrom=start, sampto=end, channels=[ch]).p_signal
                tmpdata = np.array(tmpdata).squeeze()
                tmplabel = wfdb.rdann(data_dir + '/' + name, 'atr', sampfrom=start, sampto=end).symbol
                # print(tmpdata.shape, len(tmplabel))
                
                if len(tmplabel) == 1:
                    tmplabel = tmplabel[0]
                    
                    if tmplabel in extract_labels and label_cnt_dict[tmplabel] < max_cnt:
                        label_val = label_dict[tmplabel]

                        savedata.append(tmpdata)
                        savelabels.append(label_val)

                        label_cnt_dict[tmplabel] = label_cnt_dict[tmplabel] + 1
            

        print(name, len(savedata), len(savelabels), Counter(savelabels))
        # break
    savedata = np.array(savedata)
    savelabels = np.array(savelabels)
    print('finish all', savedata.shape, Counter(savelabels))
    pd.DataFrame(savedata).to_csv(save_dir + '/data_holter.csv', index=False, header=False)
    pd.DataFrame(savelabels).to_csv(save_dir + '/labels_holter.csv', index=False, header=False)


def extraction_european(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = [i[:-4] for i in os.listdir(data_dir) if '.atr' in i]
    # print(filenames)

    # labels = []
    # for name in filenames:
    #     anno = wfdb.rdann(data_dir + '/' + name, 'atr')
    #     labels.extend((anno.symbol))
    # labels = Counter(labels)

    # for lab in extract_labels:
    #     if lab not in labels:
    #         print('Error! Not found label:', lab)

    # vals = [i for i in range(len(extract_labels))]
    # label_dict = dict(zip(extract_labels, vals))
    # print(label_dict)

    # import sys
    # sys.exit()

    savedata = []
    savelabels = []
    for name in filenames:
        # record = wfdb.rdheader(data_dir + '/' + name)
        # display(record.__dict__)
        # sys.exit()

        alldata = wfdb.rdrecord(data_dir + '/' + name).p_signal
        length, channels  = alldata.shape
        print('length', length, 'channels', channels)

        if max_idx > length:
            max_idx = length
        elif max_idx < radius * 2 + 1:
            max_idx = length
        anno = wfdb.rdann(data_dir + '/' + name, 'atr')
        peaks = anno.sample
        
        for peak in peaks:
            start = peak - radius
            end = peak + radius
            if start < 0 or end > max_idx:
                continue

            for ch in range(channels):
                if ch not in extract_channels:
                    continue

                tmpdata = wfdb.rdrecord(data_dir + '/' + name, sampfrom=start, sampto=end, channels=[ch]).p_signal
                tmpdata = np.array(tmpdata).squeeze()
                tmplabel = wfdb.rdann(data_dir + '/' + name, 'atr', sampfrom=start, sampto=end).symbol
                # print(tmpdata.shape, len(tmplabel))
                
                if len(tmplabel) == 1:
                    tmplabel = tmplabel[0]
                    
                    if tmplabel in extract_labels and label_cnt_dict[tmplabel] < max_cnt:
                        label_val = label_dict[tmplabel]

                        savedata.append(tmpdata)
                        savelabels.append(label_val)

                        label_cnt_dict[tmplabel] = label_cnt_dict[tmplabel] + 1
            

        print(name, len(savedata), len(savelabels), Counter(savelabels))
        # break
    savedata = np.array(savedata)
    savelabels = np.array(savelabels)
    print('finish all', savedata.shape, Counter(savelabels))
    pd.DataFrame(savedata).to_csv(save_dir + '/data_european.csv', index=False, header=False)
    pd.DataFrame(savelabels).to_csv(save_dir + '/labels_european.csv', index=False, header=False)


if __name__ == '__main__':

    # extract_channels = [0, 1]
    # radius = 128
    # max_idx = 650000
    # # max_idx = 360 * 60 * 30
    # extract_labels = ['N', 'L', 'R', 'V', 'A', 'B', 's', 'T']
    # data_dir = './mit-bih-arrhythmia-database-1.0.0'
    # save_dir = './data'
    # extraction_mitbih(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx)

    # extract_channels = [0, 1]
    # radius = 128
    # max_idx = 12580000
    # # max_idx = 250 * 60 * 30
    # extract_labels = ['N', 'L', 'R', 'V', 'A', 'B', 's', 'T']
    # data_dir = './sudden-cardiac-death-holter-database-1.0.0'
    # save_dir = './data'
    # extraction_holter(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx)

    extract_channels = [0, 1]
    radius = 128
    max_idx = 1800000
    # max_idx = 250 * 60 * 30
    extract_labels = ['N', 'L', 'R', 'V', 'A', 'B', 's', 'T']
    data_dir = './european-st-t-database-1.0.0'
    save_dir = './data'
    extraction_european(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx)
