from beamsearch import *

data = get_data('./dataset.arff')

searcher = BeamSearch()
search = searcher.search(data)
