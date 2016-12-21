from loaders.load_tsv import *

meta_data = load_meta_data()
click_data = load_click_data()
exp_details = load_experiment_details()

click_id = {}
exp_id = {}
meta_id = {}
user_id = {}
for row in click_data.x:
    click_id[row[4]] = None
for row in exp_details.x:
    exp_id[row[0]]= None
for row in meta_data.x:
    meta_id[row[7]] = None

for key in click_id:
    if key in exp_id and key in meta_id:
        user_id[key] = None
for key in user_id:
    print(key)