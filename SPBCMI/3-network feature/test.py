from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

nltk.download('punkt')

# 读取关系对数据，假设第一列是节点1，第二列是节点2
df = pd.read_csv('9905pairsNum.csv', header=None)
print(df)
# 合并节点1和节点2的关系
edges = list(zip(df[0], df[1]))

# 构建图形的邻接列表表示
graph = {}
for edge in edges:
    node1, node2 = edge
    if node1 not in graph:
        graph[node1] = []
    if node2 not in graph:
        graph[node2] = []
    graph[node1].append(node2)
    graph[node2].append(node1)

# 分词和预处理
tokenized_corpus = [word_tokenize(str(node)) for node in graph]

# 使用Word2Vec训练模型
model = Word2Vec(tokenized_corpus, size=64, window=5, min_count=1, sg=1, workers=1)

# 提取每个节点的特征表示
node_features = {}
for node in graph:
    node_features[node] = model.wv[str(node)]

# 将节点特征和节点名放在一起
node_features_df = pd.DataFrame.from_dict(node_features, orient='index')
node_features_df.index.name = 'node'

# 保存节点特征和节点名的DataFrame
node_features_df.to_csv('deepwalk_feature.csv', header=True)