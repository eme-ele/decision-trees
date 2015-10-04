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
        dot_product = sample.dot(self.weights)
        if dot_product >= self.tita:
            return 1
        else:
            return -1

    def promote(self, sample):
        row, col = sample.nonzero()
        for i in xrange(col.shape[0]):
            self.weights[col[i]] = self.alfa*self.weights[col[i]]

    def demote(self, sample):
        row, col = sample.nonzero()
        for i in xrange(col.shape[0]):
            self.weights[col[i]] = self.weights[col[i]]/self.alfa

    def fit(self, samples, labels):
        self.tita = samples.shape[1]
        self.init_weights(samples.shape[1])

        for i in xrange(self.max_iter):
            num_mistakes = 0

            for j in xrange(len(labels)):
                x = samples.getrow(j)
                y_pred = self.predict_one(x)

                if labels[j] == 1 and y_pred == -1:
                    num_mistakes += 1
                    self.promote(x)
                elif labels[j] == -1 and y_pred == 1:
                    num_mistakes += 1
                    self.demote(x)

            print i, " - # mistakes", num_mistakes
            if num_mistakes == 0:
                return

        return

    def predict(self, samples):
        return [self.predict_one(s) for s in samples]
