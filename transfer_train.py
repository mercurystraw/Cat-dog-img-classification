from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys

import torchvision
from tqdm import tqdm
import torch
import torch.nn as nn

from data_utils import load_data



device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('config.json') as config_file:
    config = json.load(config_file)

# Load raw data
trainset, testset = load_data(config)


# Load in the datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=2)

# Model
# 如果模型类型是resnet50，则加载预训练好的ResNet50模型，并且修改最后的全连接层将输入大小2048映射为输出2，适用于二分类任务
net = torchvision.models.resnet50(pretrained=True)
net.fc = nn.Sequential(
    nn.Linear(2048, 2)
)

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
# 学习速率调整器，每30个epoch学习率减半

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader, 0), total=len(trainloader), smoothing=0.9):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    scheduler.step()

    acc = 100. * correct / total
    print('Train ACC: %.3f' % acc)
    print('Train Loss: %.8f' % train_loss)

    return net, acc, train_loss


# Test
def eval_clean():
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader, 0), total=len(testloader), smoothing=0.9):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Test ACC: %.3f' % acc)

    return acc

accs_dir = f"trained_results/{config['MODEL_TYPE']}_epoch{config['EPOCH']}_accs"
os.makedirs(accs_dir, exist_ok=True)
accs_save_path = os.path.join(accs_dir, "train_results.txt")
with open(accs_save_path, 'w') as f:
    for epoch in range(config['EPOCH']):
            model_clean, train_acc, train_loss = train(epoch)
            test_acc = eval_clean()
            f.write(f"Epoch: {epoch}, Train ACC: {train_acc:.3f}, Test ACC: {test_acc:.3f}, Train Loss: {train_loss:.8f}\n")

model_save_path = f"trained_models/pretrained_Resnet50_epoch{config['EPOCH']}.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model_clean.state_dict(), model_save_path)

print('Train accuracy: %.3f' % train_acc)
print('Clean test accuracy: %.3f' % test_acc)
