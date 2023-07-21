import pandas as pd


def process_kmer(csv_file, input_column, k, output_column):
    # 从CSV文件中读取数据
    df = pd.read_csv(csv_file)

    # 获取指定列的内容
    input_data = df[input_column]

    # 对每一行进行判断和处理
    processed_data = []
    for element in input_data:
        if len(element) > 2000:
            # 如果序列长度超过2000，只取头部500个和后部1500个
            element = element[:500] + element[-1500:]

        # 进行kmer处理
        kmers = [element[i:i + k] for i in range(len(element) - k + 1)]

        if len(kmers) > 512:
            # 如果kmer数量超过512，只取前128个和后328个进行拼接
            kmers = kmers[:128] + kmers[-328:]

        # 添加特殊符号到处理后的序列的头部和尾部
        kmers = ['[CLS]'] + kmers + ['[SEP]']

        # 将处理后的序列用空格分隔并保存到processed_data列表中
        processed_data.append(' '.join(kmers))

    # 将处理后的结果保存在新的一列
    df[output_column] = processed_data

    # 将结果写入新的文本文件
    txt_file = 'final_sequence.txt'
    with open(txt_file, 'w') as f:
        for data in processed_data:
            f.write(data + '\n')





# 示例用法
process_kmer('all_node.csv', 'Processed_sequence', 4, 'final_column')



