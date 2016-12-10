from beamsearch import *

data = load_data('./dataset.arff', True, True)
searcher = BeamSearch()
searcher.search(data.x, data.y, data.categorical, data.attributes)
