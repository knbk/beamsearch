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

* Parameters for `run.py`:
```
usage: run.py [-h] [-m METRIC] [-w WIDTH] [-d DEPTH] [-q Q] [-v VERBOSE]
              [-t TARGET] [--minimize]

optional arguments:
  -h, --help            show this help message and exit
  -m METRIC, --metric METRIC
                        Set the metric to evaluate the subgroups. Choices are
                        'weighted_relative_accuracy', 'specificity',
                        'sensitivity', 'correlation' or 'chi_square'.
  -w WIDTH, --width WIDTH
                        Set the width of the search
  -d DEPTH, --depth DEPTH
                        Set the depth of the search.
  -q Q                  Set the number of results.
  -v VERBOSE, --verbose VERBOSE
                        Set the verbosity of the output.
  -t TARGET, --target TARGET
                        The target to measure. Choices are 'match', 'decision'
                        or 'decision_o'.
  --minimize            Minimize the measure to get the worst subgroups.
  ```
