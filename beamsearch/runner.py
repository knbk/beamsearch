from argparse import ArgumentParser

from beamsearch.beam_search import BeamSearch
from beamsearch.load_arff import get_data


class SearchRunner(object):
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument(
            '-m', '--metric',
            help="Set the metric to evaluate the subgroups. Choices are 'weighted_relative_accuracy', 'specificity', "
                 "'sensitivity', 'correlation' or 'chi_square'."
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
        parser.add_argument(
            '-v', '--verbose', type=int, default=2,
            help='Set the verbosity of the output.'
        )
        parser.add_argument(
            '-t', '--target', default='match',
            help="The target to measure. Choices are 'match', 'decision' or 'decision_o'."
        )
        parser.add_argument(
            '--minimize', default=False, action='store_true',
            help="Minimize the measure to get the worst subgroups."
        )
        parser.add_argument(
            '--min-size', default=0.1, type=float,
            help="Minimum relative size of subgroups. Should be a float in the [0.0, 1.0] range."
        )
        self.parser = parser

    def run_from_argv(self, argv):
        options = self.parser.parse_args(argv[1:])
        cmd_options = vars(options)
        search = BeamSearch(**cmd_options)
        print("Loading data...")
        data = get_data()
        print("Starting search...")
        results = search.search(data)
        print("Finished search:")
        for r in results:
            print(r)
