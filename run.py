#!/usr/bin/env python
import sys

from beamsearch.runner import SearchRunner


if __name__ == '__main__':
    runner = SearchRunner()
    runner.run_from_argv(sys.argv)
