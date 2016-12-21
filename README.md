# BeamSearch algorithm for Web Analytics

To run this:

* Install the dependencies from `requirements.txt`. 
* From the main directory (containing `requirements.txt` and `README.md`) open up a python shell.
* Run the following:

```
>>> from beamsearch import *
>>> from loaders.load_tsv import *
>>> data = load_processed_data()
>>> search = BeamSearch(metric=ttest, width=10, depth=2, q=20)
>>> results = search.search(data)
```

The `results` will be a list, with each item containing the measurement, the size of the subgroups and the descriptor conditions defining the subgroup. 
