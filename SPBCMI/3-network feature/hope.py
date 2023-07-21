import numpy as np
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import pandas as pd

# 读取关系对文件，将其转化为图形表示形式
def read_edgelist(filename):
    df = pd.read_csv(filename, header=None)
    graph = nx.from_pandas_edgelist(df, source=0, target=1)
    return graph

# 使用HOPE算法获得节点嵌入向量
def hope_embedding(graph, dimensions):
    adjacency_matrix = nx.to_numpy_matrix(graph)
    laplacian_matrix = nx.laplacian_matrix(graph)
    svd = TruncatedSVD(n_components=dimensions)
    transformed = svd.fit_transform(laplacian_matrix)
    embedding = normalize(transformed)
    return embedding

# 加载关系对文件并创建图形
filename = '9905pairsNum.csv'  # 替换为你的关系对文件路径
graph = read_edgelist(filename)

# 提取节点的嵌入向量
dimensions = 64
node_embeddings = hope_embedding(graph, dimensions)

# 保存嵌入向量到CSV文件
node_labels = list(graph.nodes)
embedding_df = pd.DataFrame(node_embeddings, columns=[f'Embedding_{i+1}' for i in range(dimensions)])
embedding_df['Node'] = node_labels
embedding_df = embedding_df[['Node'] + [f'Embedding_{i+1}' for i in range(dimensions)]]
embedding_df.to_csv('hope_feature.csv', index=False)