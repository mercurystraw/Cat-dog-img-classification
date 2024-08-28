import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
# from sklearn.model_selection import KFold
from torch.utils.data import random_split, DataLoader

from dataset import CustomDataset


def load_data(config):
    # 定义基础变换，确保所有图像尺寸一致
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为224x224，训练时使用的图像尺寸大多为224*224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像 ，选择的通用均值0.5
    ])

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    dataset = CustomDataset(root='./dataset/A',base_transform=base_transform)
    # 训练集比例0.8，测试集比例0.2，随机选择
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # 应用不同的变换
    trainset.dataset.transform = transform_train
    testset.dataset.transform = transform_test

    return trainset, testset

# def load_data_with_kfold(config, k=5):
#     transform_train = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.ToTensor()
#     ])
#     transform_test = transforms.Compose([transforms.ToTensor()])
#
#     dataset = CustomDataset(root='./dataset/A', transform=transform_train)
#     kf = KFold(n_splits=k)
#     train_test_splits = []
#     for train_index, test_index in kf.split(dataset):
#         train_subset = torch.utils.data.Subset(dataset, train_index)
#         test_subset = torch.utils.data.Subset(dataset, test_index)
#         trainloader = DataLoader(train_subset, batch_size=config['BATCH_SIZE'], shuffle=True)
#         testloader = DataLoader(test_subset, batch_size=config['BATCH_SIZE'], shuffle=False)
#         train_test_splits.append((trainloader, testloader))
#
#     return train_test_splits
#
# def change_label(dataset, config):
#
#     if config['DATASET'] in ['cifar10', 'cifar100', 'fmnist', 'mnist']:
#         # ind_0 = [i for i, label in enumerate(dataset.targets) if label == config['C0']]
#         # ind_1 = [i for i, label in enumerate(dataset.targets) if label == config['C1']]
#         # dataset.targets[ind_0] = 0
#         # dataset.targets[ind_1] = 1
#         ind_0 = [i for i, label in enumerate(dataset.targets) if label == config['C0']]
#         ind_1 = [i for i, label in enumerate(dataset.targets) if label in config['SUPER_C0']]
#         dataset.targets[ind_0] = 0
#         dataset.targets[ind_1] = 1
#     elif config['DATASET'] == 'stl10':
#         ind_0 = [i for i, label in enumerate(dataset.labels) if label in config['SUPER_C0']]
#         ind_1 = [i for i, label in enumerate(dataset.labels) if label in config['SUPER_C1']]
#         dataset.labels[ind_0] = 0
#         dataset.labels[ind_1] = 1
#
#     return dataset
