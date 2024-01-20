import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST
from tqdm import tqdm

# 导入模型
from model import LeNet_5


# 辅助函数，创建训练数据集的文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


# 重构 DataLoader
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# 处理数据集，划分为训练集和验证集
def train_valid_split(dataset, resize: tuple, train_ratio=0.8, batch_size=128,
                      shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True):
    # 下载数据集
    Train_Dataset = dataset(root='../../../../Datasets',
                            train=True,
                            transform=transforms.Compose([transforms.Resize(resize), transforms.ToTensor()]),
                            download=True)

    # 随机划分训练集和验证集
    train_size = int(train_ratio * len(Train_Dataset))
    valid_size = len(Train_Dataset) - train_size
    train_data, valid_data = Data.random_split(Train_Dataset, [train_size, valid_size])

    # 将训练集和验证集转换为可迭代的 DataLoader 对象
    train_dataloader = MultiEpochsDataLoader(dataset=train_data,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             persistent_workers=persistent_workers)

    valid_dataloader = MultiEpochsDataLoader(dataset=valid_data,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             persistent_workers=persistent_workers)
    return train_dataloader, valid_dataloader


# 训练模型
def train_model(model, train_dataloader, valid_dataloader, num_epochs, learning_rate, dataset_name):
    """
    @param model: 模型名称
    @param train_dataloader: 训练集数据
    @param valid_dataloader: 验证集数据
    @param num_epochs: 训练轮数
    @param learning_rate: 学习率
    @return: None
    """

    # 决定使用 GPU 还是 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 将模型加载到 device 当中
    model = model.to(device)
    # 复制当前模型参数作为最优模型参数
    best_model_params = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最佳验证集准确率
    best_valid_acc = 0.0
    # 分轮次训练集损失函数列表
    train_loss_list = []
    # 分轮次验证集损失函数列表
    valid_loss_list = []
    # 分轮次训练集准确率列表
    train_acc_list = []
    # 分轮次验证集准确率列表
    valid_acc_list = []
    # 当前时间
    train_start_time = time.time()

    # 分轮次训练模型
    for epoch in range(num_epochs):
        # 记录本轮次开始的的时间
        epoch_start_time = time.time()

        # 初始化每轮训练的损失值和准确率
        train_loss = 0.0
        train_corrects = 0.0
        # 初始化每轮验证的损失值和准确率
        valid_loss = 0.0
        valid_corrects = 0.0

        # 每轮训练、验证的样本数
        train_sample_num = 0
        valid_sample_num = 0

        # 对每一个 mini-batch 进行分批次训练和计算
        with tqdm(total=len(train_dataloader) + len(valid_dataloader), colour="green", ncols=100,
                  unit=' batch') as pbar:
            # 设置进度条的前缀
            pbar.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
            pbar.set_postfix_str(
                "TrainAcc:{:.2f}%  ValidAcc:{:.2f}%".format(0, 0))

            # 对每一个 mini-batch 进行分批次训练和计算
            for step, (batch_image, batch_label) in enumerate(train_dataloader):
                # 将特征图和标签数据加载到 device 上
                batch_image = batch_image.to(device)
                batch_label = batch_label.to(device)
                # 将模型设置为训练模式
                model.train()

                # 前向传播
                # 输入为一个 batch 的四维张量，大小为 (batch_size, 1, 28, 28)
                # 输出为一个 batch 的二维张量，大小为 (batch_size, 10)，表示每个样本属于 10 个类别的概率
                output = model(batch_image)
                # 计算当前训练批次的损失值
                batch_loss = criterion(output, batch_label)
                # 对于 batch_size 中每个样本找到最大的 Softmax 概率值的行号对应的标签作为预测标签
                predict_label = torch.argmax(output, dim=1)

                # 对每一批次数据，将梯度初始化为 0 再训练，防止上一批次的梯度影响当前批次的训练
                optimizer.zero_grad()
                # 反向传播
                batch_loss.backward()
                # 更新参数
                optimizer.step()

                # 将当前训练批次的损失值按照 batch_size 加权累加到当前轮次的总损失 train_loss 上
                train_loss += batch_loss.item() * batch_image.size(0)
                # 将当前训练批次的准确数量按照 batch_size 加权累加到当前轮次的总准确数 train_corrects 上
                train_corrects += torch.sum(torch.eq(predict_label, batch_label.data)).item()

                # 更新当前训练样本数
                train_sample_num += batch_image.size(0)

                # 设置进度条的监测量
                pbar.set_postfix_str(
                    "TrainAcc:{:.2f}%  ValidAcc:{:.2f}%".format(
                        round(train_corrects / train_sample_num, 4) * 100, 0))
                # 更新进度条
                pbar.update(1)

            # 对每一个 mini-batch 进行分批次验证和计算
            for step, (batch_image, batch_label) in enumerate(valid_dataloader):
                # 将特征图和标签数据加载到 device 上
                batch_image = batch_image.to(device)
                batch_label = batch_label.to(device)

                # 将模型设置为验证模式
                model.eval()

                # 前向传播计算结果，输出为一个 batch 的二维张量，大小为 (batch_size, 10)
                # 表示每个样本属于 10 个类别的概率
                output = model(batch_image)
                # 计算当前验证批次的损失值
                batch_loss = criterion(output, batch_label)
                # 对于 batch_size 中每个样本找到最大的 Softmax 概率值的行号对应的标签作为预测标签
                predict_label = torch.argmax(output, dim=1)

                # 将当前验证批次的损失值按照 batch_size 加权累加到当前轮次的验证总损失 valid_loss 上
                valid_loss += batch_loss.item() * batch_image.size(0)
                # 将当前验证批次的准确数量按照 batch_size 加权累加到当前轮次的总验证准确数 valid_corrects 上
                valid_corrects += torch.sum(torch.eq(predict_label, batch_label.data)).item()

                # 更新当前验证样本数
                valid_sample_num += batch_image.size(0)

                # 设置进度条的监测量
                pbar.set_postfix_str(
                    "TrainAcc:{:.2f}%  ValidAcc:{:.2f}%".format(
                        round(train_corrects / train_sample_num, 4) * 100,
                        round(valid_corrects / valid_sample_num, 4) * 100))
                # 更新进度条
                pbar.update(1)

        # 计算当前轮次训练的平均损失值并添加到 train_loss_list 中
        train_loss = train_loss / train_sample_num
        train_loss = round(train_loss, 4)
        train_loss_list.append(train_loss)
        # 计算当前轮次训练的平均准确率并添加到 train_acc_list 中
        train_acc = train_corrects / train_sample_num
        train_acc = round(train_acc, 4)
        train_acc_list.append(train_acc)

        # 计算当前轮次验证的平均损失值并添加到 valid_loss_list 中
        valid_loss = valid_loss / valid_sample_num
        valid_loss = round(valid_loss, 4)
        valid_loss_list.append(valid_loss)
        # 计算当前轮次验证的平均准确率并添加到 valid_acc_list 中
        valid_acc = valid_corrects / valid_sample_num
        valid_acc = round(valid_acc, 4)
        valid_acc_list.append(valid_acc)

        # 如果当前轮次验证准确率更高，则更新最佳验证准确率和最佳模型参数
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_params = copy.deepcopy(model.state_dict())

        # 打印当前轮次训练时间
        time_elapsed = time.time() - epoch_start_time
        print('Epoch {} complete in {:.0f}mim {:.0f}s'.format(epoch + 1,
                                                              time_elapsed // 60, time_elapsed % 60))

    # 为这个数据集创建一个独立的文件夹，用来记录训练过程以及最终模型
    mkdir('../output/{}'.format(dataset_name))

    # 训练结束, 保存模型
    torch.save(best_model_params, '../output/{}/best_model.pth'.format(dataset_name))

    # 将训练过程中的损失值和准确率保存为 DataFrame
    train_process = pd.DataFrame(
        data={
            'Epoch': np.arange(1, num_epochs + 1),
            'Train_Loss': train_loss_list,
            'Valid_Loss': valid_loss_list,
            'Train_Acc': train_acc_list,
            'Valid_Acc': valid_acc_list
        }
    )
    train_process.to_csv('../output/{}/train_process.csv'.format(dataset_name), index=False)

    # 打印训练总时间
    print("=" * 70)
    time_elapsed = time.time() - train_start_time
    print('Train complete in {:.0f}mim {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 返回训练过程的 DataFrame
    return train_process


# 绘制训练过程中的损失值和准确率曲线
def plot_train_process(train_process: pd.DataFrame, dataset_name: str):
    """
    @param train_process: 训练过程的 DataFrame
    @return: None
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['Epoch'], train_process['Train_Loss'], 'r-', label='Train Loss')
    plt.plot(train_process['Epoch'], train_process['Valid_Loss'], 'b', label='Valid Loss')
    plt.xlabel('Epoch')
    # 设置横坐标刻度从 0 开始，一共显示10个刻度
    plt.xticks(np.linspace(0, train_process['Epoch'].max(), 11))
    plt.ylabel('Loss')
    # 设置纵坐标刻度从 0 开始, 步长为 0.5
    plt.yticks(np.arange(0, train_process['Train_Loss'].max() + 0.5, 0.5))
    plt.legend()

    # 绘制训练集和验证集的准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['Epoch'], train_process['Train_Acc'], 'r-', label='Train Acc')
    plt.plot(train_process['Epoch'], train_process['Valid_Acc'], 'b', label='Valid Acc')
    plt.xlabel('Epoch')
    # 设置横坐标刻度从 0 开始，一共显示10个刻度
    plt.xticks(np.linspace(0, train_process['Epoch'].max(), 11))
    plt.ylabel('Acc')
    # 设置纵坐标刻度从 0% 到 100%，步长为 20%，要求格式化显示为百分数
    plt.yticks(np.arange(0, 1.2, 0.2), ['{}%'.format(int(x * 100)) for x in np.arange(0, 1.2, 0.2)])
    plt.legend()

    plt.savefig('../output/{}/train_process.png'.format(dataset_name))
    # plt.show()


if __name__ == '__main__':
    # 允许输入命令行参数
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定数据集和参数
    parser.add_argument('-d', '--dataset',
                        type=str,
                        default='FashionMNIST',
                        help='dataset name')
    # 指定训练轮次
    parser.add_argument('-e', '--num_epochs',
                        type=int,
                        default=20,
                        help='number of epochs')
    # 指定学习率
    parser.add_argument('-l', '--learning_rate',
                        type=float,
                        default=0.001,
                        help='learning rate')
    # 指定 batch_size
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=150,
                        help='batch size')
    # 指定线程数
    parser.add_argument('-w', '--num_workers',
                        type=int,
                        default=8,
                        help='number of workers')

    # 获取命令行参数
    args = parser.parse_args()
    # 设置命令行参数与数据集的映射关系
    dataset_map = {
        'FashionMNIST': FashionMNIST,
        'MNIST': MNIST
    }
    # 根据命令行参数获取数据集
    dataset = dataset_map[args.dataset]
    dataset_name = args.dataset
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_workers = args.num_workers

    # 处理数据集，划分为训练集和验证集
    train_dataloader, valid_dataloader = train_valid_split(dataset, resize=(28, 28),
                                                           train_ratio=0.8, batch_size=batch_size,
                                                           shuffle=True, num_workers=num_workers,
                                                           pin_memory=True, persistent_workers=True)

    # 实例化模型
    model = LeNet_5()

    # 分布式训练
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 训练模型
    train_process = train_model(model, train_dataloader, valid_dataloader,
                                num_epochs=num_epochs, learning_rate=learning_rate,
                                dataset_name=dataset_name)

    # 绘制训练过程中的损失值和准确率曲线
    plot_train_process(train_process, dataset_name=dataset_name)
