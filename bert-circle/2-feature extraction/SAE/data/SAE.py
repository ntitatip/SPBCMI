import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
import torch
import csv
import torch
import torch.nn as nn
import torch.optim as optim


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

model = Word2Vec(result, size=50, window=2, min_count=0, workers=12)

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

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

input_data = padded_sentences
input_data = input_data.view(-1, 507*25)

# 定义自编码器模型
autoencoder = Autoencoder(input_size=507*25, hidden_size=64)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=0.000001)

# 进行训练
num_epochs = 20
for epoch in range(num_epochs):
    # 前向传播
    encoded, decoded = autoencoder(input_data)

    # 计算重构损失
    loss = criterion(decoded, input_data)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 提取特征
encoded_features, _ = autoencoder(input_data)  # 获取编码器的输出
encoded_features = encoded_features.view(3308, -1)  # 将特征表示重新调整为3308个样本
print("Encoded Features Size:", encoded_features.size())

array = encoded_features.detach().numpy()

# 保存NumPy数组到CSV文件
filename = 'SAE_feature.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)  # 写入列标题
    writer.writerows(array)      # 写入数据行