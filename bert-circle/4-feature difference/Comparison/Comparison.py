from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from matplotlib.font_manager import FontProperties
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import os


class DataMerger:
    def __init__(self):
        self.dictionaries = {}

    def add_to_dictionary(self, dictionary_name, key, value):
        if dictionary_name not in self.dictionaries:
            self.dictionaries[dictionary_name] = {}
        self.dictionaries[dictionary_name][key] = value


    def read_csv_data_df(self, csv_file):
        data = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame({'Node1': data[0], 'Node2': data[1]}, index=range(len(data)))
        self.add_to_dictionary('Train_list', 'positive_samples', df)
        return df

    def generate_negative_samples(self):
        positive_samples = self.dictionaries['Train_list']['positive_samples']
        all_nodes = set(positive_samples['Node1']).union(set(positive_samples['Node2']))
        num_positive_samples = len(positive_samples)
        negative_samples = pd.DataFrame(columns=['Node1', 'Node2'])
        df = positive_samples

        while len(negative_samples) < 2*num_positive_samples:
            nodes = np.random.choice(list(all_nodes), size=2, replace=False)
            node1 = nodes[0]
            node2 = nodes[1]

            if any(((df['Node1'] == node1) & (df['Node2'] == node2)) |
                   ((df['Node1'] == node2) & (df['Node2'] == node1))):
                continue

            if any(((negative_samples['Node1'] == node1) & (negative_samples['Node2'] == node2)) |
                   ((negative_samples['Node1'] == node2) & (negative_samples['Node2'] == node1))):
                continue

            if node1 == node2:
                continue

            negative_samples = negative_samples.append({'Node1': node1, 'Node2': node2}, ignore_index=True)
        self.add_to_dictionary('Train_list', 'negative_samples', negative_samples)
        return negative_samples

    def read_csv_with_header(self, csv_file):
        data = {}
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                feature_name = row[0]
                features = np.array(row[1:], dtype=np.float32)
                data[feature_name] = features
        df = pd.DataFrame(data)
        df_T = df.T
        csv_name = os.path.basename(csv_file)
        feature_type = os.path.splitext(csv_name)[0]
        self.add_to_dictionary('Features', f'{feature_type}', df_T)

    def process_multiple_csv_files(self, csv_files):
        for csv_file in csv_files:
            self.read_csv_with_header(csv_file)

    def process_mulitiple_features(self):
        for feature_name, feature_array in self.dictionaries['Features'].items():
            Traindata = self.merge_features(self.dictionaries['Train_list']['train_list'], feature_array)
            self.add_to_dictionary('Train_Data_with_features', feature_name, Traindata)

    def merge_features(self, df, feature_array):
        # 重置DataFrame的索引
        df = df.reset_index(drop=True)

        # 创建一个空的二维数组，用于保存合并后的特征
        merged_features = np.empty((len(df), feature_array.shape[1] * 2))

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
        merged_df = pd.DataFrame(merged_features, columns=[f"Feature_{i}" for i in range(feature_array.shape[1] * 2)])

        # 将合并特征的DataFrame与原始DataFrame合并
        merged_df = pd.concat([df, merged_df], axis=1)

        return merged_df

    def generate_train_list(self):
        positive_samples = self.dictionaries['Train_list']['positive_samples']
        negative_samples = self.generate_negative_samples()
        positive_samples["Label"] = 1
        negative_samples["Label"] = 0
        merged_df = pd.concat([positive_samples, negative_samples], axis=0)
        merged_df = merged_df.reset_index(drop=True)
        self.add_to_dictionary('Train_list', 'train_list', merged_df)

    def generate_trainlist_with_features(self, pair_csv, feature_csv):
        self.read_csv_data_df(pair_csv)
        self.process_multiple_csv_files(feature_csv)
        self.generate_train_list()
        self.process_mulitiple_features()
        Trainlist_with_features = self.dictionaries['Train_Data_with_features']

        subfolder = 'traindata'
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        for key, df in Trainlist_with_features.items():
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            file_path = os.path.join(subfolder, f'TrainData_with_features_{timestamp}_{key}.csv')
            df.to_csv(file_path, index=False)
            print("DataFrame 已保存到子文件夹:", file_path)
        return Trainlist_with_features




# 创建 DataMerger 实例

data_merger = DataMerger()

# 读取 pair 文件
csv_files = ['Combined Features.csv', 'Structural Features.csv', 'Node Features.csv']
filename = '9905pairsNum.csv'
TrainData_with_features = data_merger.generate_trainlist_with_features(filename, csv_files)

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.dfs = TrainData_with_features

    def train_random_forest(self, X, y, name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.model
        model.fit(X_train, y_train)

        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        precision, recall, _ = precision_recall_curve(y_test, y_scores)

        # 计算评估指标
        acc = accuracy_score(y_test, y_scores.round())
        mcc = matthews_corrcoef(y_test, y_scores.round())
        auc = roc_auc_score(y_test, y_scores)
        aupr = average_precision_score(y_test, y_scores)
        f1 = f1_score(y_test, y_scores.round())
        evaluation_matrix = np.array([[acc, mcc, auc, aupr, f1, name]])


        return evaluation_matrix, fpr, tpr, precision, recall

    def plot_auc_aupr_curves(self):
        fpr_list = []
        tpr_list = []
        precision_list = []
        recall_list = []
        evaluations = []

        for i, df in self.dfs.items():
            X = df.drop(['Label', 'Node1', 'Node2'], axis=1)
            y = df['Label']
            evaluation_matrix, fpr, tpr, precision, recall = self.train_random_forest(X, y, i)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            precision_list.append(precision)
            recall_list.append(recall)
            evaluations.append(evaluation_matrix)

        array1 = np.squeeze(evaluations)
        df = pd.DataFrame(array1, columns=['acc', 'mcc', 'auc', 'aupr', 'f1', 'name'])

        subfolder_evaluations = 'evaluations'

        # 保存DataFrame到子文件夹
        folder_path = './' + subfolder_evaluations + '/'  # 子文件夹路径
        file_path = folder_path + 'evaluations.csv'  # 保存文件路径

        # 确保子文件夹存在
        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存DataFrame到文件
        df.to_csv(file_path, index=False)

        print("保存成功到子文件夹:", file_path)



        # 绘制AUC曲线
        plt.figure()
        normalized_rgb_values = []
        rgb_values = [(231, 56, 71), (168, 218, 219), (69, 123, 157), (29, 53, 87), (240, 250, 239)]
        for rgb in rgb_values:
            normalized_rgb = tuple(value / 255.0 for value in rgb)
            normalized_rgb_values.append(normalized_rgb)

        name1 = []
        for i in range(len(fpr_list)):
            acc, mcc, auc, aupr, f1, name = array1[i]
            name1.append(name)

        name_lengths = [len(name) for name in name1]

        # 找到最大的句子长度
        max_length = max(name_lengths)

        # 填充或截断句子，使其长度一致
        padded_strings = [name.ljust(max_length) for name in name1]
        print(padded_strings)

        for i in range(len(fpr_list)):
            if i == 0:
                linestyle = '-'
            else:
                linestyle = ':'
            acc, mcc, auc, aupr, f1, name = array1[i]
            plt.plot(fpr_list[i], tpr_list[i], color=normalized_rgb_values[i],
                     label=f'{padded_strings[i]} (AREA={float(auc):.4f})', linestyle=linestyle)

        legend_font = FontProperties(family='Arial Monospaced MT', size=11)
        title_font = FontProperties(family='Arial', size=14)

        line = plt.gca().lines[0]  # 获取第一根线的对象
        line.set_linewidth(2)
        line.set_zorder(10)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Comparison of AUC Curves for Independent \n and Combined features', linespacing=1.5,
                  fontproperties=title_font, pad=13)
        plt.legend(loc='lower right', prop=legend_font)

        subfolder_images = 'images'
        if not os.path.exists(subfolder_images):
            os.makedirs(subfolder_images)
            # 保存图表到子文件夹
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(subfolder_images, f'auc_{timestamp}.tif')
        plt.savefig(filename)
        plt.show()


        # 绘制AUPR曲线
        plt.figure()
        normalized_rgb_values = []
        rgb_values = [(231, 56, 71), (168, 218, 219), (69, 123, 157), (29, 53, 87), (240, 250, 239)]
        for rgb in rgb_values:
            normalized_rgb = tuple(value / 255.0 for value in rgb)
            normalized_rgb_values.append(normalized_rgb)

        name2 = []
        for i in range(len(fpr_list)):
            acc, mcc, auc, aupr, f1, name = array1[i]
            name2.append(name)

        name_lengths = [len(name) for name in name2]

        # 找到最大的句子长度
        max_length = max(name_lengths)

        # 填充或截断句子，使其长度一致
        padded_strings2 = [name.ljust(max_length) for name in name2]
        print(padded_strings2)

        lines = []
        labels = []
        auprs = []
        for i in range(len(fpr_list)):
            if i == 0:
                linestyle = '-'
            else:
                linestyle = ':'
            acc, mcc, auc, aupr, f1, name = array1[i]
            line, = plt.plot(recall_list[i], precision_list[i], color=normalized_rgb_values[i],
                             label=f'{padded_strings2[i]} (AREA={float(aupr):.4f})', linestyle=linestyle)
            lines.append(line)
            labels.append(line.get_label())
            auprs.append(aupr)

        legend_font = FontProperties(family='Arial Monospaced MT', size=11)
        title_font = FontProperties(family='Arial', size=14)

        line = plt.gca().lines[0]  # 获取第一根线的对象
        line.set_linewidth(2)
        line.set_zorder(10)
        plt.plot([0, 1], [1, 0], 'k--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Comparison of AUPR Curves for Independent \n and Combined features',
                  linespacing=1.5, fontproperties=title_font, pad=13)
        plt.legend(bbox_to_anchor=(0.0, 0), loc='lower left', prop=legend_font)

        subfolder = 'images'
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        # 保存图表到子文件夹

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(subfolder, f'aupr_{timestamp}.tif')
        plt.savefig(filename)
        plt.show()



model = GradientBoostingClassifier(n_estimators=20, max_depth=2)
model_trainer = ModelTrainer(model)
model_trainer.plot_auc_aupr_curves()
