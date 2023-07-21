import csv


def convert_rna_to_dna_from_csv(input_file, input_column, output_column):
    """从 CSV 文件中读取指定列的 RNA 序列，将其转换为 DNA 序列，并保存到 CSV 文件的新列"""
    with open(input_file, 'r', newline='') as csv_input:
        reader = csv.DictReader(csv_input)
        headers = reader.fieldnames
        headers.append(output_column)

        output_rows = []
        for row in reader:
            rna_sequence = row[input_column]
            dna_sequence = convert_rnaUtoT(rna_sequence)
            row[output_column] = dna_sequence
            output_rows.append(row)

    with open(input_file, 'w', newline='') as csv_output:
        writer = csv.DictWriter(csv_output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)


def convert_rnaUtoT(sequence):
    """将 RNA 序列中的 U 替换为 T，然后将其转换为 DNA 序列"""
    return sequence.replace('U', 'T')


# 示例使用
input_file = 'mirna_convertUtoT.csv'
input_column = 'sequence'
output_column = 'DNA_sequence'

convert_rna_to_dna_from_csv(input_file, input_column, output_column)
