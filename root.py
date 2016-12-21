from beamsearch import *
from discrimination_search import find_discrimination
from loaders import *

meta_data = load_meta_data()
clicking_data = load_click_data()
experiment_details = load_experiment_details()

meta_data.set_target_index(3)
meta_data.set_target_index(4)

for i in range(meta_data.x.shape[1]):
    print(meta_data.attributes[i] + ": " + str(meta_data.x[:10, i]))