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

model = Word2Vec(result, size=100, window=5, min_count=1, workers=4)

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


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        out, hn = self.gru(x, h0)  # 前向传播，获取GRU的输出和最终隐藏状态
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出特征

        return out, hn

# 创建GRU模型实例

x = torch.Tensor(padded_sentences)

input_size = 100  # 输入特征的大小
hidden_size = 20  # 隐藏层大小
num_layers = 2  # GRU层数
output_size = 5  # 输出特征的大小

model = GRUModel(input_size, hidden_size, num_layers, output_size)

# 准备输入数据，假设张量的维度为（batch_size, sequence_length, input_size）

outputs, hn = model(x)

# 获取最后一个时间步的输出特征
last_output = outputs[:, -1, :]

print(last_output.shape)  # 输出特征的形状

# 打印输出张量的形状
array = last_output.detach().numpy()

# 保存NumPy数组到CSV文件
filename = 'GRU_feature.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)  # 写入列标题
    writer.writerows(array)