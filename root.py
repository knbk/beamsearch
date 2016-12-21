from beamsearch import *
from discrimination_search import find_discrimination
from loaders import *

meta_data = load_meta_data()
# clicking_data = load_click_data()
# experiment_details = load_experiment_details()

meta_data.encode_values()
