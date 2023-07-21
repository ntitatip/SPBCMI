import pandas as pd
import networkx as nx
import numpy as np
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 读取关系对文件，将其转化为图形表示形式
def read_edgelist(filename):
    df = pd.read_csv(filename, header=None)
    graph = nx.from_pandas_edgelist(df, source=0, target=1)
    return graph

# 创建自定义数据集
class Node2VecDataset(Dataset):
    def __init__(self, graph):
        self.graph = graph
        self.node_list = list(graph.nodes)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, index):
        node = self.node_list[index]
        return torch.tensor(node)

# 定义node2vec模型
class Node2Vec(nn.Module):
    def __init__(self, embedding_dim, num_nodes):
        super(Node2Vec, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, node):
        embedding = self.embedding(node)
        hidden = self.linear1(embedding)
        output = self.linear2(hidden)
        return output

# 设置训练参数
embedding_dim = 64
num_epochs = 100
batch_size = 64
learning_rate = 0.001

# 加载关系对文件并创建图形
filename = '9905pairsNum.csv'  # 替换为你的关系对文件路径
graph = read_edgelist(filename)

# 创建数据集和数据加载器
dataset = Node2VecDataset(graph)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = Node2Vec(embedding_dim, len(graph.nodes))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模型训练
for epoch in range(num_epochs):
    for nodes in dataloader:
        optimizer.zero_grad()
        nodes = nodes.squeeze()
        embeddings = model(nodes)
        loss = criterion(embeddings, model.embedding.weight[nodes])
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 提取节点的嵌入表示和对应的节点标识
node_embeddings = model.embedding.weight.detach().numpy()
node_labels = list(graph.nodes)

# 创建包含节点标识和嵌入向量的DataFrame
embeddings_df = pd.DataFrame({'Node': node_labels})
for i in range(embedding_dim):
    embeddings_df[f'Embedding_{i+1}'] = node_embeddings[:, i]

# 将嵌入表示保存到CSV文件
embeddings_df.to_csv('node2vec_feature.csv', index=False)