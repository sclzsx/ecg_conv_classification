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
from matplotlib import pyplot as plt
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
# print(label_dict)

label_dict_decode = dict(zip(vals, extract_labels))

vals2 = [0 for i in range(len(extract_labels))]
label_cnt_dict = dict(zip(extract_labels, vals2))
# print(label_cnt_dict)


max_cnt = 5000


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
        length, channels = alldata.shape
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

                    # if tmplabel in extract_labels:
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
        length, channels = alldata.shape
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

                    # if tmplabel in extract_labels:
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
        length, channels = alldata.shape
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

                    # if tmplabel in extract_labels:
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


def split_train_val():
    data1 = np.array(pd.read_csv('./data/data_european.csv'))
    label1 = np.array(pd.read_csv('./data/labels_european.csv'))

    data2 = np.array(pd.read_csv('./data/data_holter.csv'))
    label2 = np.array(pd.read_csv('./data/labels_holter.csv'))

    data3 = np.array(pd.read_csv('./data/data_mitbih.csv'))
    label3 = np.array(pd.read_csv('./data/labels_mitbih.csv'))

    print(data1.shape, label1.shape)
    print(data2.shape, label2.shape)
    print(data3.shape, label3.shape)

    data = np.r_[data1, data2, data3]
    label = np.r_[label1, label2, label3]
    print(data.shape, label.shape)

    min_ = np.min(data)
    max_ = np.max(data)
    data = (data - min_) / (max_ - min_)

    print(Counter(label.flatten()))
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, label, test_size=0.2, random_state=21, stratify=label)
    print(Counter(train_labels.flatten()))
    print(Counter(test_labels.flatten()))

    pd.DataFrame(train_data).to_csv('./data/train_data.csv', index=False, header=False)
    pd.DataFrame(test_data).to_csv('./data/test_data.csv', index=False, header=False)
    pd.DataFrame(train_labels).to_csv('./data/train_labels.csv', index=False, header=False)
    pd.DataFrame(test_labels).to_csv('./data/test_labels.csv', index=False, header=False)


def data_transform(data, augment):
    def scaling(X, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
        myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
        return X * myNoise

    def verflip(sig):
        b = sig[:, ::-1]
        return b

    def shift(sig, interval=20):
        for col in range(sig.shape[1]):
            offset = np.random.choice(range(-interval, interval))
            sig[:, col] += offset
        return sig

    if augment:
        data = np.expand_dims(data, 0)
        if np.random.randn() > 0.5: data = scaling(data)
        if np.random.randn() > 0.5: data = verflip(data)
        if np.random.randn() > 0.5: data = shift(data)
        data = data.squeeze(0)

    return data


class ECGDataset(Dataset):
    def __init__(self, data, labels, augment):
        self.data = data
        self.labels = labels
        self.num = len(data)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = data_transform(self.data[i], self.augment)
        data = torch.from_numpy(np.ascontiguousarray(data).astype('float32')).unsqueeze(0)
        # data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # required shape [N 1 250]

        target = self.labels[i]
        target = torch.from_numpy(np.ascontiguousarray(target).astype('int64')).squeeze()
        # target = torch.tensor(target, dtype=torch.int64)  # 分类时，为了适应交叉熵函数，需要转成long

        return data, target


if __name__ == '__main__':
    do_extraction = 0
    do_split = 0

    if do_extraction:
        extract_channels = [0, 1]
        radius = 128
        # max_idx = 650000
        max_idx = 360 * 60 * 30
        extract_labels = ['N', 'L', 'R', 'V', 'A', 'B', 's', 'T']
        data_dir = './mit-bih-arrhythmia-database-1.0.0'
        save_dir = './data'
        extraction_mitbih(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx)

        extract_channels = [0, 1]
        radius = 128
        # max_idx = 12580000
        max_idx = 250 * 60 * 30
        extract_labels = ['N', 'L', 'R', 'V', 'A', 'B', 's', 'T']
        data_dir = './sudden-cardiac-death-holter-database-1.0.0'
        save_dir = './data'
        extraction_holter(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx)

        extract_channels = [0, 1]
        radius = 128
        # max_idx = 1800000
        max_idx = 250 * 60 * 30
        extract_labels = ['N', 'L', 'R', 'V', 'A', 'B', 's', 'T']
        data_dir = './european-st-t-database-1.0.0'
        save_dir = './data'
        extraction_european(data_dir, save_dir, radius, extract_channels, extract_labels, max_idx)

    if do_split:
        split_train_val()

    train_data = np.array(pd.read_csv('./data/train_data.csv'))
    train_labels = np.array(pd.read_csv('./data/train_labels.csv'))

    print(Counter(train_labels.flatten()))

    train_dataset = ECGDataset(data=train_data, labels=train_labels, augment=False)
    for i, j in train_dataset:
        print(i.shape, j.shape, j, i.dtype, j.dtype)

        plt.plot(np.arange(256), np.squeeze(i))
        plt.show()

        break
