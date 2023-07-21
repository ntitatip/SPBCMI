import csv

def reverse_transcribe_from_csv(input_file, input_column, output_column):
    """从 CSV 文件中读取指定列的序列数据，将其逆转录为 DNA，并保存到 CSV 文件的新列"""
    base_pairs = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    with open(input_file, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames
        headers.append(output_column)

        output_rows = []
        for row in reader:
            sequence = row[input_column]
            dna_sequence = reverse_transcribe(sequence, base_pairs)
            row[output_column] = dna_sequence
            output_rows.append(row)

    with open(input_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)

def reverse_transcribe(sequence, base_pairs):
    """将序列进行逆转录，将碱基反向互补为 DNA 序列"""
    dna_sequence = ''
    for base in reversed(sequence):
        if base in base_pairs:
            dna_sequence += base_pairs[base]
        else:
            dna_sequence += base
    return dna_sequence

# 示例使用
input_file = 'all_node.csv'
input_column = 'RNA_sequence'
output_column = 'DNA_sequence'

reverse_transcribe_from_csv(input_file, input_column, output_column)

