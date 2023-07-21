# 假设你已经有一个包含所有序列信息的列表 sequences

# 创建空的 k-mer 到索引的映射字典

import numpy as np
import torch
import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
import torch
import csv


def process_text_file(file_path):
    word_list = []  # 用于保存转换后的结果

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("[CLS]"):
                line = line[5:]

                # 去除结尾的 "sep" 符号
            if line.endswith("[SEP]"):
                line = line[:-5]

            line = line.replace(' ', ',')

            # 将转换后的内容分割成列表
            line_words = line.split(',')

            # 移除空字符串元素（如果有的话）
            line_words = list(filter(None, line_words))

            # 将每行的结果添加到总的列表中
            word_list.append(line_words)

    return word_list
# 遍历每个序列

kmer_map = {}
def indices_onehot(sequences):
    index = 0
    for sequence in sequences:
        # 遍历序列中的每个 k-mer
        for kmer in sequence:
           # 如果 k-mer 不在映射字典中，则为其分配一个索引
            if kmer not in kmer_map:
                kmer_map[kmer] = index
                index += 1
    return kmer_map

file_path = 'final_sequence.txt'  # 替换为你实际的文件路径
kmer_sequence = process_text_file(file_path)
# print(kmer_sequence)
kmers = indices_onehot(kmer_sequence)
# print(kmers)

# 创建 One-Hot 编码矩阵

def one_hot_encoding(sequence, kmer_index_map):
    kmer_count = len(kmer_index_map)
    encoded_sequence = np.zeros((len(sequence), kmer_count))

    for i, kmer in enumerate(sequence):
        index = kmer_index_map[kmer]
        encoded_sequence[i, index] = 1

    return encoded_sequence

# 假设你已经有了包含3308条序列的列表 sequences
# 假设你已经将所有的 k-mer 进行了索引，并保存在字典 kmer_index_map 中

one_hot_sequences = []

sequences = kmer_sequence
kmer_index_map = kmers
sequence_tensor = []
for sequence in sequences:
    encoded_sequence = one_hot_encoding(sequence, kmer_index_map)
    sequence_tensor.append(encoded_sequence)

sentence_lengths = [len(sentence) for sentence in sequence_tensor]
max_length = max(sentence_lengths)

padded_sentences = pad_sequence([torch.tensor(sentence, dtype=torch.float32) for sentence in sequence_tensor], batch_first=True)
print(padded_sentences.shape)


class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=1, stride=2)

        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=1, stride=2)

        # 第三层卷积
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=1, stride=2)

        # 第四层卷积
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 64)  # 输入维度为32 * 252，输出维度为64

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度顺序以适应卷积层的输入
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc(x)
        return x

model_1 = FeatureExtractorCNN()


# 定义损失函数和优化器（不需要优化器，因为我们不进行训练）
criterion = nn.MSELoss()
with torch.no_grad():  # 禁用梯度计算，因为我们不需要进行反向传播和优化
    features = model_1(padded_sentences)

# 输出特征形状
array = features.numpy()


# 定义损失函数和优化器（不需要优化器，因为我们不进行训练）
filename = 'CNN_feature.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)  # 写入列标题
    writer.writerows(array)
