from loaders.load_tsv import *

reader = load_meta_data()
url_count = {}
city_count = {}
i = 0
for row in reader.x:
    if i > 1000:
        break
    i += 1
    url_count.setdefault(row[13], 0)
    url_count[row[13]] += 1
    city_count.setdefault(row[12], 0)
    city_count[row[12]] += 1

for w in sorted(url_count, key=url_count.get, reverse=True):
    print(url_count[w],w)

for w in sorted(city_count, key=city_count.get, reverse=True):
    print(city_count[w],w)