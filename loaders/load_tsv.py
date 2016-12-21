import numpy as np
from .datamodel import DataModel
import csv


def load_experiment_details():
    return create_data_model(load_tsv('dataset/experiment_details'))


def load_click_data():
    return create_data_model(load_tsv('dataset/clicking_data'))


def load_meta_data():
    part1 = load_tsv('dataset/meta_data_1')
    part2 = load_tsv('dataset/meta_data_2')
    return create_data_model(np.concatenate((part1, part2)))


def create_data_model(data):
    attributes = data[1, :]
    return DataModel(data[1:, :], np.array([]), attributes=attributes)


def load_tsv(path):
    lines = []
    with open(path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            lines.append(row)
    return np.array(lines, dtype=object)
