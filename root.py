from beamsearch import *
from discrimination_search import find_discrimination
from loaders import load_tsv

meta_data_1 = load_tsv('dataset/meta_data_1')
for line in meta_data_1: