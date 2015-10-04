import numpy as np
import time

class perceptron(object):

    def __init__(self, max_iter, learn_rate):
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.weights = np.array([])
        self.weight_0 = 0 #bias

    # weights should be set to random small values
    def init_weights(self, num_feats):
        self.weight_0 = 0
        self.weights = np.zeros(num_feats)

    def sgn(self, sample):
        dot_product = 0
        #print sample
        for elem in sample.iteritems():
            i,j = elem[0]
            dot_product += sample[i,j] * self.weights[j]
        #dot_product = sample.dot(self.weights)
        ## add bias weight_0 to dot_product
        dot_product += 1*self.weight_0
        if dot_product > 0:
            return 1
        else:
            return -1

    def update_weights(self, x, y):
        # calculate bias weight_0
        self.weight_0 = self.weight_0 + self.learn_rate * y * 1
        for elem in x.iteritems():
            i,j = elem[0]
            self.weights[j] += self.learn_rate * y * x[i,j]

        #self.weights = self.weights + self.learn_rate * y * x
        #self.weights = self.weights.getA1()
        #print self.weights
        #   self.weights[i] = self.weights[i] + \
        #                      self.learn_rate * y * x[i]

    def fit(self, samples, labels):
        # init weights for all features
        self.init_weights(samples.shape[1])

        for i in xrange(self.max_iter):
            num_mistakes = 0

            for j in xrange(len(labels)):
                y_pred = self.sgn(samples[j])

                if y_pred != labels[j]:
                    num_mistakes += 1
                    self.update_weights(samples[j], labels[j])

            print i, " - # mistakes", num_mistakes

            if num_mistakes == 0:
                return
        return

    def predict(self, samples):
        return [self.sgn(s) for s in samples]
