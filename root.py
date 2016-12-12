from beamsearch import *
from daans_attempt import test_subsets

data = get_data('./dataset.arff')

searcher = BeamSearch()
subsets = searcher.search(data)
test_subsets(data, subsets)