import numpy as np
import re
from collections import Counter
import sys


class ngrams_fe(object):

    def __init__(self, i, j=None):
        self.regex = r''
        self.n_grams = set([])
        self.i = i
        if j:
            self.j = j
        else:
            self.j = i


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

        self.n_grams = set(self.union_ngrams(ss))
        print "number of n_grams", len(self.n_grams)

        '''regex_str = r''
        for ng in self.n_grams:
            for i in xrange(len(ng)):
                regex_str += r'\b{0}'.format(ng[i])
            regex_str += r'\b|'
        regex_str = regex_str[:-1]
        self.regex = re.compile(regex_str)'''

    def extract(self, samples):
        ret = []
        '''
        for i, sample in enumerate(samples):
            ret.append(Counter(self.regex.findall(sample)))
            sys.stdout.write("\r\textracted {0}% of users".\
                    format((i * 100) / len(samples)))
        sys.stdout.write('\n')'''
        for i, sample in enumerate(samples):
            sample = sample.split()
            obs_n_grams = set([])
            for x in xrange(self.i, self.j + 1):
                obs_n_grams = obs_n_grams.union(set(self.sample_ngrams(sample, x)))
            ret.append(Counter(obs_n_grams.intersection(self.n_grams)))
            sys.stdout.write("\r\textracted {0}% of users".\
                    format((i * 100) / len(samples)))
        sys.stdout.write('\n')

        return ret


