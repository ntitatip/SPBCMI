import pandas as pd


def merge_columns(csv_file1, csv_file2, column1, column2, output_file):
    # 从第一个CSV文件中读取指定列
    df1 = pd.read_csv(csv_file1)
    cols1 = df1[[column1, column2]]  # 选择第一个文件的两列

    # 从第二个CSV文件中读取指定列
    df2 = pd.read_csv(csv_file2)
    cols2 = df2[[column1, column2]]  # 选择第二个文件的两列

    # 创建一个新的DataFrame，将第二个文件的内容放在第一个文件的列下面
    merged_df = pd.concat([cols1, cols2], axis=0, ignore_index=True)

    # 将合并的结果写入新的CSV文件
    merged_df.to_csv(output_file, index=False)

    # 打印拼接后的两列长度
    print(f"第一列长度: {len(cols1)}")
    print(f"第二列长度: {len(cols2)}")

# 示例用法
merge_columns('mirna_convertUtoT.csv', '2346circ不重复带化学式-带名称.csv', 'name', 'Processed_sequence', 'all_node.csv')

