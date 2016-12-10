# BeamSearch algorithm for Web Analytics

To run this:

* Install the dependencies from `requirements.txt`. 
* From the main directory (containing `requirements.txt` and `README.md`) open up a python shell.
* Run the following:

```
>>> from beamsearch import *
>>> X, y, categorical, attributes = get_data()
>>> results = search.search(X, y, categorical, attributes)
```

* You can create a new BeamSearch instance and pass in a different metric or width/depth:
```
>>> search = BeamSearch(metric=chi_square, width=25, depth=3, q=20)
```
