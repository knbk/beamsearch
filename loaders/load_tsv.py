import numpy as np
from .datamodel import DataModel


def load_experiment_details():
    return create_datamodel(load_tsv('dataset/experiment_details'))


def load_click_data():
    return create_datamodel(load_tsv('dataset/clicking_data'))


def load_meta_data():
    part1 = load_tsv('dataset/meta_data_1')
    part2 = load_tsv('dataset/meta_data_2')
    return create_datamodel(np.concatenate((part1, part2)))


def create_datamodel(data):
    attributes = data[1, :]
    return DataModel(data[1:, :], np.array([]), attributes=attributes)


def load_tsv(path):
    lines = []
    with open(path, 'r') as file:
        line = file.readline()
        while line != '':
            lines.append(line.split('\t'))
            line = file.readline()
    print('rows: ' + str(len(lines)))
    print('columns: ' + str(len(lines[1])))
    return np.array(lines, dtype=object)
