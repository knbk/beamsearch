import sys

from beamsearch.runner import SearchRunner


runner = SearchRunner()
runner.run_from_argv(sys.argv)
