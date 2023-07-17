import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
import torch
import csv
# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 假设你的特征向量为（3308，50700），将其转换为Tensor
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

X_flattened = padded_sentences.reshape(3308, 507*100)

# 定义自编码器模型，假设输入维度为50700，编码维度为64
input_dim = 50700
encoding_dim = 64
autoencoder = Autoencoder(input_dim, encoding_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练自编码器
num_epochs = 10
for epoch in range(num_epochs):
    # 前向传播
    outputs = autoencoder(X_flattened)
    loss = criterion(outputs, X_flattened)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失值
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 提取特征
encoded_features = autoencoder.encoder(X_flattened)
print('Encoded features shape:', encoded_features.shape)

array = encoded_features.detach().numpy()

filename = 'AE_feature.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)  # 写入列标题
    writer.writerows(array)      # 写入数据行
