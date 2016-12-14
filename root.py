from beamsearch import *
from discrimination_search import find_discrimination

data = get_data('./dataset.arff')

searcher = BeamSearch(metric=semi_elift, q=40)
subsets = searcher.search(data)
print('x')
with open('discrimination_subgroups.csv', 'w') as file:
    file.write("measure;count;attributes\n")
    for i in range(len(subsets)):
        subset = subsets[i]
        file.write(str(subset.measure) + ";" + str(subset.count) + ";" + str(subset.attributes).replace(',', ' and ') + "\n")
# find_discrimination(data)