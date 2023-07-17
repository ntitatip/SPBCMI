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


# 使用示例
file_path = 'final_sequence.txt'  # 替换为你实际的文件路径
result = process_text_file(file_path)

model = Word2Vec(result, size=100, window=5, min_count=1, workers=12)

# 保存模型
model.save("word2vec.model")

# 加载保存的模型
loaded_model = Word2Vec.load("word2vec.model")

# 使用模型进行词向量嵌入
embedded_sentences = []

for sentence in result:
    # 嵌入句子中的每个词
    embedded_words = []
    for word in sentence:
        if word in model.wv:
            word_vector = model.wv[word]
            embedded_words.append(word_vector)

    # 将嵌入结果添加到保存的数据结构中
    embedded_sentences.append(embedded_words)

print(len(embedded_sentences))

# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc = nn.Linear(32 * 56 * 56, 64)  # 全连接层
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.pool1(out)
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.pool2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# 假设你已经定义好了一个嵌入后的句子列表 embedded_sentences

sentence_lengths = [len(sentence) for sentence in embedded_sentences]

# 找到最大的句子长度
max_length = max(sentence_lengths)

# 填充或截断句子，使其长度一致
padded_sentences = pad_sequence([torch.tensor(sentence) for sentence in embedded_sentences], batch_first=True)
print(padded_sentences.shape)
# 打印特征的形状

class RNNFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        features = self.fc(hidden)
        return features

# 定义输入数据的形状
input_size = 100  # 输入特征的维度
hidden_size = 16  # RNN隐藏单元的维度
output_size = 64  # 输出特征的维度

# 创建RNN模型实例
rnn_model = RNNFeatureExtractor(input_size, hidden_size, output_size)

# 创建RNN模型实例

features = rnn_model(padded_sentences)

print(features.shape)

array = features.detach().numpy()

# 保存NumPy数组到CSV文件
filename = 'RNN_feature.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)  # 写入列标题
    writer.writerows(array)