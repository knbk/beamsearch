import numpy as np
import csv


def load_experiment_details():
    return load_tsv('dataset/experiment_details')


def load_click_data():
    return load_tsv('dataset/clicking_data')


def load_meta_data():
    part1 = load_tsv('dataset/meta_data_1')
    part2 = load_tsv('dataset/meta_data_2')
    return np.concatenate((part1, part2))


def load_tsv(path):
    lines = []
    with open(path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            lines.append(row)
    print('rows: ' + str(len(lines)))
    print('columns: ' + str(len(lines[1])))
    return np.array(lines, dtype=object)
