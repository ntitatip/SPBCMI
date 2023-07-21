import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx

# 读取关系对数据，假设第一列是节点1，第二列是节点2
df = pd.read_csv('9905pairsNum.csv', header=None)

# 创建一个无向图
G = nx.Graph()
edges = list(zip(df[0], df[1]))
G.add_edges_from(edges)

# 构建邻接矩阵（对称矩阵）
adj_matrix = nx.adjacency_matrix(G)
adj_matrix = adj_matrix.toarray()
adj_matrix = np.maximum(adj_matrix, adj_matrix.T)  # 将邻接矩阵设置为对称矩阵

# 设置参数
num_nodes = adj_matrix.shape[0]
input_dim = num_nodes
hidden_dim1 = 256
hidden_dim2 = 128
output_dim = 64
learning_rate = 0.01
num_epochs = 20

# 定义SDNE模型
class SDNE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SDNE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 将邻接矩阵转换为PyTorch Tensor
adj_matrix = torch.from_numpy(adj_matrix).float()

# 创建SDNE模型
model = SDNE(input_dim, hidden_dim1, hidden_dim2, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    optimizer.zero_grad()
    encoded, decoded = model(adj_matrix)
    loss = criterion(decoded, adj_matrix)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 获取节点的嵌入向量
with torch.no_grad():
    encoded, _ = model(adj_matrix)
node_embeddings = encoded.numpy()

# 将节点嵌入向量和节点名放在一起
node_embeddings_df = pd.DataFrame(node_embeddings, columns=[f"dim_{i+1}" for i in range(output_dim)])
node_embeddings_df['node'] = G.nodes()

# 保存节点嵌入向量和节点名的DataFrame
node_embeddings_df.to_csv('sdne_feature.csv', index=False)

