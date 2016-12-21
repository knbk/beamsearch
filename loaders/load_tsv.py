from collections import defaultdict

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


def to_timestamp(val):
    min, sec = val.split(':')
    return 60 * float(min) + float(sec)


def load_processed_data():
    metadata = load_meta_data()
    experiment_data = load_experiment_details()
    experiments = defaultdict(list)
    for row in experiment_data.x:
        experiments[row[0]].append((row[2], to_timestamp(row[3])))
    # Cut off anchor from url, so ping url matches view url
    metadata.x[:, 13] = [x.split('#')[0] for x in metadata.x[:, 13]]
    # Convert string timestamps to seconds (in float)
    metadata.x[:, 1] = [to_timestamp(x) for x in metadata.x[:, 1]]
    metadata.x[:, 2] = [to_timestamp(x) for x in metadata.x[:, 2]]
    metadata.x[:, 3] = [to_timestamp(x) for x in metadata.x[:, 3]]
    view_data = defaultdict(list)
    ping_data = defaultdict(list)
    for x in metadata.x:
        if x[4] == 'page_view':
            view_data[(x[7], x[13])].append(x)
        elif x[4] == 'page_ping':
            ping_data[(x[7], x[13])].append(x)
    # sort on dvce_created_tstamp
    for v in view_data.values():
        v.sort(key=lambda x: x[3])
    view_time = {}
    view_dict = {}
    for k, v in view_data.items():
        for i, row in enumerate(v):
            view_dict[k + (i,)] = row
            if i < len(v) - 1:
                pings = [x for x in ping_data[(row[7], row[13])] if row[3] <= x[3] < v[i + 1][3]]
                end_time = max([p[3] for p in pings]) if pings else row[3]
            else:
                pings = [x for x in ping_data[(row[7], row[13])] if row[3] <= x[3]]
                end_time = max([p[3] for p in pings]) if pings else row[3]
            t = end_time - row[3] if end_time > 0 else 0.0
            view_time[(row[7], row[13], i)] = t

    for k, v in view_time.items():
        key, index = k[:2], k[2]
        view = view_data[key][index]
        view_data[key][index] = np.zeros((len(view) + 2), dtype=object)
        view_data[key][index][:-2] = view
        view_data[key][index][-2] = v
        max_dist = 999
        group = None
        for row in experiments[key[0]]:
            dist = abs(view_dict[k][3] - row[1])
            if dist < max_dist:
                max_dist = dist
                group = row[0]
        view_data[key][index][-1] = group

    metadata.x = np.array([x for v in view_data.values() for x in v])
    metadata.attributes = np.concatenate((metadata.attributes, ['visit_duration', 'experiment_group']))
    return metadata


def create_data_model(data):
    attributes = data[0, :]
    attribute_types = data[1, :]
    return DataModel(data[2:, :], np.array([]), attributes=attributes, attribute_types=attribute_types)


def load_tsv(path):
    lines = []
    with open(path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            lines.append(row)
    return np.array(lines, dtype=object)
