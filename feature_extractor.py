import numpy as np
import re
from collections import Counter
import sys
import itertools


class ngrams_fe(object):

    def __init__(self, i, j=None):
        self.i = i
        if j:
            self.j = j
        else:
            self.j = i
        self.n_gram_dict = {}

    def sample_ngrams(self, sample, n):
        return zip(*[sample[x:] for x in range(n)])

    def union_ngrams(self, samples):
        n_grams = []
        for index in xrange(self.i, self.j+1):
            for s in samples:
                n_grams += self.sample_ngrams(s, index)
        return n_grams

    def train(self, samples, labels):
        ss = samples[:]
        ss = [s.split() for s in samples]
        ss = [filter(lambda x: x.isalnum(), s) for s in ss]

        n_grams = set(self.union_ngrams(ss))
        positions = [i for i in xrange(len(n_grams))]

        self.n_grams_dict = dict(itertools.izip(n_grams, positions))

        print "number of n_grams", len(n_grams)

    def extract(self, samples):
        feat_set = set(self.n_grams_dict.keys())
        ret = np.zeros(shape=(len(samples), len(feat_set)))

        for i, sample in enumerate(samples):
            sample = sample.split()
            obs_n_grams = set([])

            for x in xrange(self.i, self.j + 1):
                obs_n_grams = obs_n_grams.union(set(self.sample_ngrams(sample, x)))

            out = obs_n_grams.intersection(feat_set)
            for e in out:
                ret[i][self.n_grams_dict[e]] = 1.0

            sys.stdout.write("\r\textracted {0}% of users".\
                    format((i * 100) / len(samples)))

        sys.stdout.write('\n')

        return ret
