import csv
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model_path = 'pytorch_model.bin'
config_path = 'config.json'
tokenizer_path = 'C:/Users/Jiren_ZHOU/Downloads/vocab.txt'
config = BertConfig.from_json_file(config_path)
model = BertModel.from_pretrained(model_path, config=config).to(device)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# 设置全连接层
fc = torch.nn.Linear(config.hidden_size, 64).to(device)
tanh = torch.nn.Tanh()

# 读取CSV文件
csv_path = 'input.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

# 对每500个样本进行特征提取
batch_size = 500
features = []
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    texts = [row[0] for row in batch]

    # 将文本转化为BERT输入
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        # 得到BERT输出
        outputs = model(**inputs)

        # 提取特征
        text_embeddings = outputs[0][:, 0, :]
        features.append(text_embeddings)

# 合并所有特征，并通过全连接层得到最终特征
features = torch.cat(features, dim=0)
features = fc(features)
features = tanh(features)

# 将特征保存到CSV文件
output_path = 'feature1.csv'
with open(output_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for feature in features.cpu().numpy():
        writer.writerow(feature)