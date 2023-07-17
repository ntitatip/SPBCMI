import torch
import torch.nn as nn
import csv
import numpy as np

class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv1d(100, 16, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4000, 64)  # 输入维度为32 * 252，输出维度为64

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度顺序以适应卷积层的输入
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc(x)
        return x

model = FeatureExtractorCNN()

# 创建随机输入数据
batch_size = 3308
sequence_length = 507
input_dim = 100
inputs = torch.randn(batch_size, sequence_length, input_dim)

# 提取特征
with torch.no_grad():
    features = model(inputs)

# 输出特征形状
# print('Features shape:', features.shape)
# 保存NumPy数组到CSV文件
filename = 'tensor.csv'
array = features.numpy()

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(array)      # 写入数据行
