import pandas as pd
from sklearn.decomposition import PCA

# 读取节点特征数据
node_features_df = pd.read_csv('node_features1.csv', header=None)
print(node_features_df)

# 提取特征矩阵
features = node_features_df.drop(node_features_df.columns[0], axis=1).values
# print(features)

# 创建PCA模型，设置要降至的维度
pca = PCA(n_components=16)

# 在特征矩阵上进行PCA降维
features_pca = pca.fit_transform(features)
features_pca_df = pd.DataFrame(features_pca)
# print(features_pca_df)

# 将节点名和降维后的特征合并到一起
final_df = pd.concat([node_features_df.iloc[:, 0], features_pca_df], axis=1)
print(final_df)
# 保存降维后的特征结果
final_df.to_csv('pca_features.csv', index=False)