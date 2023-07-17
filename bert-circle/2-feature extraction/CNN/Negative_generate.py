import csv
import numpy as np
import pandas as pd
# import random
from datetime import datetime

from scipy.interpolate import interp1d
from sklearn.model_selection import cross_val_score, KFold, train_test_split, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc


def read_csv_data(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # if len(row) == 2:  # 仅处理包含两个元素的行
            data.append(row)
    return data


def read_csv_with_header(csv_file):
    data = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # header = next(reader)  # 读取标题行并忽略
        for row in reader:
            feature_name = row[0]
            features = np.array(row[1:], dtype=np.float32)  # 转换为浮点数类型的NumPy数组
            data[feature_name] = features
        df = pd.DataFrame(data)
        df_T = df.T
    return df_T


def read_csv_data_df(csv_file):
    data = pd.read_csv(csv_file, header=None)
    df = pd.DataFrame({'Node1': data[0], 'Node2': data[1]}, index=range(len(data)))
    return df


def merge_features(df, feature_array):
    # 重置DataFrame的索引
    df = df.reset_index(drop=True)

    # 创建一个空的二维数组，用于保存合并后的特征
    merged_features = np.empty((len(df), feature_array.shape[1]*2))

    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        # 获取节点1和节点2的索引，并转换为整数类型
        node1_index = row['Node1']
        node2_index = row['Node2']
        # 根据索引从特征数组中按行获取对应的特征向量
        node1_features = feature_array.iloc[node1_index]
        node2_features = feature_array.iloc[node2_index]

        # 合并两个向量作为边的特征
        edge_features = np.concatenate((node1_features, node2_features))

        # 将合并后的特征添加到二维数组中
        merged_features[index] = edge_features

    # 创建包含合并特征的DataFrame
    merged_df = pd.DataFrame(merged_features, columns=[f"Feature_{i}" for i in range(feature_array.shape[1]*2)])

    # 将合并特征的DataFrame与原始DataFrame合并
    merged_df = pd.concat([df, merged_df], axis=1)

    return merged_df


def generate_negative_samples(positive_samples):
    # 获取正样本中的所有节点
    all_nodes = set(positive_samples['Node1']).union(set(positive_samples['Node2']))

    # 获取正样本的数量
    num_positive_samples = len(positive_samples)

    # 创建一个空的负样本数据容器
    negative_samples = pd.DataFrame(columns=['Node1', 'Node2'])
    df = positive_samples
    # 循环生成负样本，直到负样本数量与正样本数量一致
    while len(negative_samples) < num_positive_samples:
        # 随机选择两个节点作为负样本的两个端点
        nodes = np.random.choice(list(all_nodes), size=2, replace=False)
        node1 = nodes[0]
        node2 = nodes[1]
        # 检查负样本是否与正样本重复，以及两个节点是否相同
        if any(((df['Node1'] == node1) & (df['Node2'] == node2)) | ((df['Node1'] == node2) & (df['Node2'] == node1))):
            continue

            # 判断关系对是否与已保存的负样本重复
        if any(((negative_samples['Node1'] == node1) & (negative_samples['Node2'] == node2)) | (
                (negative_samples['Node1'] == node2) & (negative_samples['Node2'] == node1))):
            continue

        if node1 == node2:
            continue

            # 添加关系对到负样本DataFrame
        negative_samples = negative_samples.append({'Node1': node1, 'Node2': node2}, ignore_index=True)
    return negative_samples


def generate_train_dataset(positive_samples):
    # 获取正样本
    negative_samples = generate_negative_samples(pair)
    positive_samples["Label"] = 1
    negative_samples["Label"] = 0
    merged_df = pd.concat([positive_samples, negative_samples], axis=0)
    features = merge_features(merged_df, feature)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_path = f'TrainData_{timestamp}.csv'
    features.to_csv(file_path, index=False)
    return features


class CrossValidationPlot:
    def __init__(self, model):
        self.model = model
        self.cv = 5  # 五折交叉验证
        self.shuffle_split = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=42)  # 打乱数据

    def train_and_compute_metrics(self, X, y):
        metrics_list = []
        all_y_test = []
        all_y_pred = []
        all_fpr = []
        all_tpr = []
        all_precision = []
        all_recall = []

        for i, (train_index, test_index) in enumerate(self.shuffle_split.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba(X_test)[:, 1]

            # 计算指标值
            mcc = matthews_corrcoef(y_test, y_pred.round())
            acc = accuracy_score(y_test, y_pred.round())
            auc_score = roc_auc_score(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            aupr = auc(recall, precision)
            f1 = f1_score(y_test, y_pred.round())

            # 保存指标值到列表
            metrics_list.append({'Fold': i+1, 'MCC': mcc, 'ACC': acc, 'AUC': auc_score, 'AUPR': aupr, 'F1': f1})

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_fpr.append(roc_curve(y_test, y_pred)[0])
            all_tpr.append(roc_curve(y_test, y_pred)[1])
            all_precision.append(precision)
            all_recall.append(recall)

        # 计算平均指标值
        metrics_df = pd.DataFrame(metrics_list)
        mean_metrics = metrics_df.mean()
        mean_metrics['Fold'] = 'Mean'
        metrics_df = metrics_df.append(mean_metrics, ignore_index=True)

        # 保存指标值到CSV文件
        metrics_df.to_csv('metrics.csv', index=False)

        # 绘制AUC和AUPR曲线
        self.plot_auc_curve(all_fpr, all_tpr)
        self.plot_aupr_curve(all_recall, all_precision)

    def plot_auc_curve(self, all_fpr, all_tpr):
        plt.figure(figsize=(10, 6))

        interp_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.zeros_like(interp_fpr)

        for i in range(self.cv):
            fpr = all_fpr[i]
            tpr = all_tpr[i]
            auc_value = auc(all_fpr[i], all_tpr[i])
            interp_tpr += np.interp(interp_fpr, fpr, tpr)
            plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {auc_value:.4f})')

        interp_tpr /= self.cv
        mean_fpr = interp_fpr
        mean_tpr = interp_tpr
        mean_auc = np.mean([auc(all_fpr[i], all_tpr[i]) for i in range(self.cv)])
        plt.plot(mean_fpr, mean_tpr,label=f'Mean (AUC = {mean_auc:.4f})')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (AUC)')

        # 保存 AUC 曲线图
        plt.legend(loc='lower right')
        plt.savefig('auc_curve.png')
        plt.close()

    def plot_aupr_curve(self, all_recall, all_precision):
        plt.figure(figsize=(10, 6))

        for i in range(self.cv):
            recall = all_recall[i]
            precision = all_precision[i]
            aupr = np.trapz(precision, recall)
            plt.plot(recall, precision, label=f'Fold {i + 1}:  {abs(aupr):.4f}')

        # 计算平均曲线和平均AUPR值
        max_length = max([len(recall) for recall in all_recall])
        mean_recall = np.linspace(0, 1, max_length)
        mean_precision = np.zeros(max_length)
        for i in range(self.cv):
            f = interp1d(all_recall[i], all_precision[i], kind='linear')
            mean_precision += f(mean_recall)
        mean_precision /= self.cv

        mean_aupr = np.trapz(mean_precision, mean_recall)

        plt.plot(mean_recall, mean_precision, label=f'Mean (AUPR = {mean_aupr:.4f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (AUPR)')

        # 保存 AUPR 曲线图
        plt.legend(loc='lower left')
        plt.savefig('aupr_curve.png')
        plt.close()

pair = read_csv_data_df('9905pairsNum.csv')
feature = read_csv_with_header('tensor.csv')

Traindataset = generate_train_dataset(pair)
X = Traindataset.drop(['Label', 'Node1', 'Node2'], axis=1)
y = Traindataset['Label']

# 创建Random Forest分类器模型对象
model = RandomForestClassifier()

# 创建CrossValidationPlot对象并调用绘图方法
cv_plot = CrossValidationPlot(model)
cv_plot.train_and_compute_metrics(X, y)

