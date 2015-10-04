import numpy as np

class winnow(object):

    def __init__(self, max_iter, alfa=2.0):
        self.max_iter = max_iter
        self.alfa = alfa
        self.tita = 0
        self.weights = np.array([])

    def init_weights(self, num_feats):
        self.weights = np.ones(num_feats)

    def predict_one(self, sample):
        dot_product = 0
        for (j,v) in sample:
            dot_product += v * self.weights[j]
        if dot_product >= self.tita:
            return 1
        else:
            return -1

    def promote(self, sample):
        for (j,v) in sample:
            self.weights[j] = self.alfa*self.weights[j]

    def demote(self, sample):
        for (j,v) in sample:
            self.weights[j] = self.weights[j]/self.alfa

    def fit(self, samples, labels, n_feats):
        self.tita = n_feats
        self.init_weights(n_feats)

        for i in xrange(self.max_iter):
            num_mistakes = 0

            for j in xrange(len(labels)):
                y_pred = self.predict_one(samples[j])

                if labels[j] == 1 and y_pred == -1:
                    num_mistakes += 1
                    self.promote(samples[j])
                elif labels[j] == -1 and y_pred == 1:
                    num_mistakes += 1
                    self.demote(samples[j])

            print i, " - # mistakes", num_mistakes
            if num_mistakes == 0:
                return

        return

    def predict(self, samples):
        return [self.predict_one(s) for s in samples]
