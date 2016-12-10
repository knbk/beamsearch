import functools
import heapq as hq
import operator
from functools import reduce
import numpy as np
import random

from beamsearch import metrics


class BeamSearch(object):
    def __init__(self, metric=None, width=5, depth=2, q=10, bins=8):
        if metric and not callable(metric):
            metric = getattr(metrics, metric)
        self.metric = metric or metrics.weighted_relative_accuracy
        self.width = width
        self.depth = depth
        self.q = q
        self.bins = bins
        self.random = random.Random(1)

    # def run_classifier(self, X, y):
    #     results = []
    #     model = self.model
    #
    #     for fold_no, (train, test) in enumerate(self.skf.split(X, y)):
    #         trainX = X[train]
    #         trainY = y[train]
    #         testX = X[test]
    #         testY = y[test]
    #
    #         model.fit(trainX, trainY)
    #         ProbaY = model.predict_proba(testX)
    #         PredY = model.predict(testX)
    #
    #         for i in range(len(test)):
    #             line = [fold_no, test[i]]
    #             line.extend(ProbaY[i])
    #             line.append(PredY[i])
    #             line.append(testY[i])
    #             results.append(line)
    #
    #     return results

    def get_splitters(self, feature, categorical):
        res = []
        if categorical:
            fset = set(feature)
            if len(fset) < 2:
                return res
            elif len(fset) == 2:
                val = fset.pop()
                eq = functools.partial(operator.eq, val)
                eq.comparison = '=='
                res.append(eq)
                ne = functools.partial(operator.ne, val)
                ne.comparison = '!='
                res.append(ne)
            else:
                for val in set(feature):
                    eq = functools.partial(operator.eq, val)
                    eq.comparison = '=='
                    res.append(eq)
                    ne = functools.partial(operator.ne, val)
                    ne.comparison = '!='
                    res.append(ne)
        else:
            fset = set()
            for i in range(self.bins):
                # Comparison labels are swapped because the split value
                # is the first parameter and the dataset the second
                val = np.percentile(feature, i * 100.0 / self.bins)
                if val in fset:
                    continue
                le = functools.partial(operator.le, val)
                le.comparison = '>='
                res.append(le)
                ge = functools.partial(operator.ge, val)
                ge.comparison = '<='
                res.append(ge)
                fset.add(val)
        return res

    def get_subgroup(self, x, splitters):
        def merge_subgroups(sub1, splitter):
            sub2 = splitter(x[:, splitter.attr_index])
            return np.logical_and(sub1, sub2)

        return reduce(merge_subgroups, splitters, np.ones(x[:, 0].shape, np.bool_))

    def width_search(self, x, y, candidates, categorical):
        results = []

        p_subgroup = self.get_subgroup(x, candidates)

        push = hq.heappush

        for i in range(x.shape[1]):
            feature = x[:, i]
            for splitter in self.get_splitters(feature, categorical[i]):
                splitter.attr_index = i
                subgroup = np.logical_and(p_subgroup, splitter(x[:, splitter.attr_index]))
                measure = self.metric(x, y, subgroup)
                push(results, (measure, self.random.random(), candidates + [splitter]))
                if len(results) >= self.width:
                    push = hq.heappushpop

        return results

    def depth_search(self, x, y, categorical):
        results = []

        def get_all(x):
            return np.ones(x.shape, np.bool_)

        get_all.attr_index = 0

        candidates = [[get_all]]

        for d in range(self.depth):
            beam = []
            while candidates:
                cds = candidates.pop()
                width_search = self.width_search(x, y, cds, categorical)
                beam = hq.nlargest(self.width, hq.merge(beam, width_search))
                results = hq.nlargest(self.q, hq.merge(results, width_search))

            candidates = [cds for i, r, cds in beam]

        return [(measure, cds) for measure, _, cds in results]

    def get_splitter_descr(self, splitter, attributes):
        attr = attributes[splitter.attr_index]
        if isinstance(attr[1], list):
            val = attr[1][int(splitter.args[0])]
        else:
            val = splitter.args[0]
        return '%s %s %s' % (attr[0], splitter.comparison, val)

    def search(self, data):
        results = self.depth_search(data.x, data.y, data.categorical)
        new_res = []
        for measure, candidates in results:
            newline = [measure, np.count_nonzero(self.get_subgroup(data.x, candidates))]
            for c in candidates[1:]:
                newline.append(self.get_splitter_descr(c, data.attributes))
            new_res.append(newline)

        return new_res


search = BeamSearch()
