import pandas as pd

def process_text_file(file_path):
    word_list = []  # 用于保存转换后的结果

    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.replace(' ', ',')

            # 将转换后的内容分割成列表
            line_words = line.split(',')

            # 移除空字符串元素（如果有的话）
            line_words = list(filter(None, line_words))

            # 将每行的结果添加到总的列表中
            word_list.append(line_words)

    return word_list


input_file = ['9905deepwalk64.txt', '9905hope64.txt', '9905line64.txt', '9905node2vec64.txt', '9905sdne64.txt']


def processed_csv(texts, name):
    df = pd.DataFrame(texts)
    new_df = df.iloc[1:]
    new_df.reset_index(drop=True, inplace=True)
    # new_df.set_index(new_df.columns[0])
    df_sorted = new_df.sort_values(by=new_df.columns[0], key=lambda x: x.astype(int))

    print(df_sorted)
    df_sorted.to_csv(f'{name}.csv', index=False, header=False)


for text in input_file:
    feature_name = text[:-4]
    raw_data = process_text_file(text)
    processed_csv(raw_data, feature_name)

