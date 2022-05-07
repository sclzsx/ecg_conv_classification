import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import ECGDataset, extract_labels, label_dict_decode
from vgg import VGG
from resnet import RESNET
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择用cpu还是gpu

batch_size = 64
max_epoch = 50

model_id = 0


def get_labels_weight(labels, method=0):
    distribute = dict(Counter(labels.flatten()))

    num_classes = len(distribute)

    num_labels = [i for i in distribute.values()]
    max_num_labels = max(num_labels)

    weight = np.zeros(num_classes)

    for i in range(num_classes):
        if method == 0:
            weight[i] = 1 / distribute[i]
        else:
            weight[i] = max_num_labels / distribute[i]

    return weight


def train():
    if model_id == 0:
        model = VGG().to(device)
        save_dir = 'results/VGG/'
    else:
        model = RESNET().to(device)
        save_dir = 'results/RESNET/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data = np.array(pd.read_csv('./data/train_data.csv'))
    train_labels = np.array(pd.read_csv('./data/train_labels.csv'))
    print(train_data.shape, train_labels.shape)

    train_dataset = ECGDataset(data=train_data, labels=train_labels, augment=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Finish loading dataset, begin training.')

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    weight = torch.tensor(get_labels_weight(train_labels), dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    for epoch in range(1, max_epoch + 1):  # range的区间是左闭右开,所以加1

        model.train()  # 训练模式

        mean_loss = 0

        for i, (batch_data, batch_labels) in enumerate(train_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
            # print(batch_data.shape, batch_labels.shape)
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
            batch_labels = batch_labels.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式

            optimizer.zero_grad()  # 清零已有梯度

            batch_pred = model(batch_data)  # 前向传播,获得网络的输出

            batch_loss = criterion(batch_pred, batch_labels)

            batch_loss_val = batch_loss.item()

            if i % 50 == 0:
                print(epoch, i, batch_loss_val)

            mean_loss = mean_loss + batch_loss_val  # 累加所有批次的平均损失. item()的意思是取数值,因为该变量不是一个tensor

            batch_loss.backward()  # 反向传播损失,更新模型参数

            optimizer.step()  # 更新学习率

        mean_loss = mean_loss / (i + 1)  # 损失和求均,为当前epoch的损失

        log = {'epoch': epoch, 'mean loss': mean_loss}
        print('######', log)

        torch.save(model.state_dict(), save_dir + 'latest.pth')


def visualize_tensor_in_out(input, output, save_path):
    with torch.no_grad():
        input = input.squeeze().cpu()
        output = output.squeeze().cpu()

    plt.plot(input, 'b')
    plt.plot(output, 'r')
    plt.fill_between(np.arange(256), output, input, color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.savefig(save_path)
    plt.cla()


def plot_confusion_matrix(confusion, save_path):
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(extract_labels))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # axis = [str(i) for i in range(num_classes)]
    plt.xticks(indices, extract_labels)
    plt.yticks(indices, extract_labels)

    plt.colorbar()

    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            plt.text(first_index, second_index, confusion[first_index][second_index])
    # 在matlab里面可以对矩阵直接imagesc(confusion)

    plt.savefig(save_path)
    plt.cla()


def eval():
    if model_id == 0:
        model = VGG().to(device)
        save_dir = 'results/VGG/'
    else:
        model = RESNET().to(device)
        save_dir = 'results/RESNET/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_data = np.array(pd.read_csv('./data/test_data.csv'))
    test_labels = np.squeeze(np.array(pd.read_csv('./data/test_labels.csv')))
    test_dataset = ECGDataset(data=test_data, labels=test_labels, augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Finish loading dataset, begin testing.')

    model_path = save_dir + 'latest.pth'
    model.load_state_dict(torch.load(model_path))

    all_labels = []
    all_preds = []
    for i, (batch_data, batch_labels) in enumerate(test_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
        with torch.no_grad():
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
            batch_pred = model(batch_data)  # 前向传播,获得网络的输出

            preds = batch_pred.squeeze().cpu().numpy().argmax(axis=1)  # 依次进行: 降维,转为cpu张量,转为np,求每一行最大值的索引
            labels = batch_labels.squeeze().cpu().numpy()

            all_labels.extend(list(labels))
            all_preds.extend(list(preds))

    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='weighted')
    r = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf = confusion_matrix(all_labels, all_preds)
    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1}
    print(metrics)

    plot_confusion_matrix(conf, save_dir + 'confusion_matrix.png')


def demo():
    if model_id == 0:
        model = VGG().to(device)
        save_dir = 'results/VGG/'
    else:
        model = RESNET().to(device)
        save_dir = 'results/RESNET/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_data = np.array(pd.read_csv('./data/test_data.csv'))
    test_labels = np.squeeze(np.array(pd.read_csv('./data/test_labels.csv')))
    test_dataset = ECGDataset(data=test_data, labels=test_labels, augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    print('Finish loading dataset, begin testing.')

    model_path = save_dir + 'latest.pth'
    model.load_state_dict(torch.load(model_path))

    for i, (batch_data, batch_labels) in enumerate(test_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
        with torch.no_grad():
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
            batch_pred = model(batch_data)  # 前向传播,获得网络的输出

            pred = batch_pred.cpu().numpy().argmax(axis=1)  # 依次进行: 降维,转为cpu张量,转为np,求每一行最大值的索引

            lab = batch_labels.squeeze().cpu().item()
            pre = pred[0]
            if lab == pre:
                flag = True
                color = 'green'
            else:
                flag = False
                color = 'red'

            text = 'Label is :{}, Prediction is :{}, Recognized :{}'.format(label_dict_decode[lab],
                                                                            label_dict_decode[pre], flag)
            print(text)

            plt.title(text)
            plt.plot(np.arange(256), batch_data.squeeze().cpu().numpy(), color=color)
            plt.show()


if __name__ == '__main__':
    # train()

    eval()

    demo()
