from beamsearch import *
from discrimination_search import find_discrimination

data = get_data('./dataset.arff')

searcher = BeamSearch(metric=semi_elift)
subsets = searcher.search(data)
print('x')
# find_discrimination(data)