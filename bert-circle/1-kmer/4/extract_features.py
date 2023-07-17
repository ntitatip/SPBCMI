import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from torch import nn


class TextNet(nn.Module):
    def __init__(self, code_length):
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('config.json')
        self.textExtractor = BertModel.from_pretrained(
            'pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()
        self.output_layer = nn.Linear(code_length, code_length)

    def forward(self, tokens, segments, input_masks):
        with torch.no_grad():
            output = self.textExtractor(tokens, token_type_ids=segments,
                                        attention_mask=input_masks)
            text_embeddings = output[0][:, 0, :]
            features = self.fc(text_embeddings)
            features = self.tanh(features)
            output = self.output_layer(features)
        return output


def process_txt(input_file, output_file, code_length, batch_size):
    tokenizer = BertTokenizer.from_pretrained('vocab.txt')

    textNet = TextNet(code_length=code_length)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    textNet = textNet.to(device)

    texts = []
    with open(input_file, 'r') as f:
        for line in f:
            texts.append(line.strip())

    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens])

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding

    tokens_tensor = torch.tensor(tokens).to(device)
    segments_tensors = torch.tensor(segments).to(device)
    input_masks_tensors = torch.tensor(input_masks).to(device)

    dataset = TensorDataset(tokens_tensor, segments_tensors, input_masks_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_text_hashCodes = []

    for i, (batch_tokens, batch_segments, batch_input_masks) in enumerate(dataloader):
        try:
            with torch.no_grad():
                batch_tokens = batch_tokens.to(device)
                batch_segments = batch_segments.to(device)
                batch_input_masks = batch_input_masks.to(device)
                batch_text_hashCodes = textNet(batch_tokens, batch_segments, batch_input_masks)
            all_text_hashCodes.append(batch_text_hashCodes)
            print(f"Batch {i+1}/{len(dataloader)} processed successfully.")
        except Exception as e:
            print(f"Batch {i+1}/{len(dataloader)} processing failed with error: {str(e)}")

    text_hashCodes = torch.cat(all_text_hashCodes, dim=0)

    numpy_array = text_hashCodes.detach().cpu().numpy()

    np.savetxt('feature_4mer.txt',numpy_array,delimiter=' ')
    np.savetxt(output_file, numpy_array, delimiter=",")



# 示例用法
input_txt_file = 'final_sequence.txt'
output_csv_file = 'feature_4mer.csv'
# k-mer长度
code_length = 64  # fc映射到的维度大小
batch_size = 64  # 批次大小

process_txt(input_txt_file, output_csv_file, code_length, batch_size)

