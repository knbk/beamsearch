from collections import Counter, defaultdict

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


def load_processed_metadata():
    metadata = load_meta_data()
    experiment_data = load_experiment_details()
    experiments = defaultdict(set)
    for row in experiment_data.x:
        experiments[row[0]].add(row[2])
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
        group = None
        if len(experiments[key[0]]) == 1:
            for val in experiments[key[0]]:
                group = val
        view_data[key][index][-1] = group

    metadata.x = np.array([x for v in view_data.values() for x in v])
    metadata.attributes = np.concatenate((metadata.attributes, ['visit_duration', 'experiment_group']))
    return metadata


def load_processed_data():
    metadata = load_processed_metadata()
    clicking_data = load_click_data()
    clicks = Counter([x[-1] for x in clicking_data.x if x[0] == 'clic'])
    views = Counter([x[-1] for x in clicking_data.x if x[0] == 'view'])
    datadict = defaultdict(list)
    for row in metadata.x:
        if row[-1] is not None:
            datadict[row[7]].append(row)

    data = {}
    for user_id, rows in datadict.items():
        mc_country = Counter([x[8].strip() for x in rows]).most_common(1)[0][0]
        mc_language = Counter([x[22].strip() for x in rows]).most_common(1)[0][0]
        num_visits = len(rows)
        avg_duration_all = sum([x[-2] for x in rows]) / len(rows)
        long_visits = [x for x in rows if x[-2] > 0]
        num_long_visits = len(long_visits)
        avg_duration = (sum([x[-2] for x in long_visits]) / len(long_visits)) if long_visits else 0
        total_duration = sum([x[-2] for x in rows])
        group = rows[0][-1]
        mc_device = Counter([x[29] for x in rows]).most_common(1)[0][0]
        referrer_name = Counter([x[16] for x in rows]).most_common(1)[0][0]
        num_clicks = clicks.get(user_id, 0)
        num_views = views.get(user_id, 0)
        line = [
            mc_country, mc_language, total_duration, num_visits, avg_duration_all, num_long_visits, avg_duration,
            referrer_name, mc_device, group, num_clicks, num_views,
        ]
        data[user_id] = np.asarray(line, dtype=object)

    model = DataModel(np.asarray(list(data.values()), dtype=object), np.array([]), attributes=[
        'most_common_country', 'most_common_language', 'total_duration', 'num_visits', 'avg_duration_all', 'num_long_visits',
        'avg_duration_long', 'referrer_name', 'most_common_device', 'group', 'num_clicks', 'num_views',
    ], attribute_types=[
        'string', 'string', 'float', 'int', 'float', 'int', 'float', 'string', 'string', 'string', 'int', 'int',
    ], categorical=[
        True, True, False, False, False, False, False, True, True, True, False, False,
    ])
    model.encode_values()
    model.set_target_index(slice(9, 12))
    return model


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
