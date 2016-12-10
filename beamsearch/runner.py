from argparse import ArgumentParser

from beamsearch.beam_search import BeamSearch
from beamsearch.load_arff import get_data


class SearchRunner(object):
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument(
            '-m', '--metric', choices=['weighted_relative_accuracy', 'specificity', 'sensitivity', 'correlation', 'chi_square'],
            help='Set the metric to evaluate the subgroups.'
        )
        parser.add_argument(
            '-w', '--width', type=int, default=5,
            help='Set the width of the search'
        )
        parser.add_argument(
            '-d', '--depth', type=int, default=2,
            help='Set the depth of the search.'
        )
        parser.add_argument(
            '-q', type=int, default=10,
            help='Set the number of results.'
        )
        self.parser = parser

    def run_from_argv(self, argv):
        options = self.parser.parse_args(argv[1:])
        cmd_options = vars(options)
        search = BeamSearch(**cmd_options)
        print("Loading data...")
        X, y, categorical, attributes = get_data()
        print("Starting search...")
        results = search.search(X, y, categorical, attributes)
        print("Finished search:")
        for r in results:
            print(r)
