import numpy as np

class perceptron(object):

    def __init__(self, max_iter, learn_rate):
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.weights = np.array([])
        self.bias = 0 #bias
        self.avg_weights = np.array([])
        self.avg_bias = np.array([])

    # weights should be set to random small values
    def init_weights(self, num_feats):
        self.bias = 0
        self.weights = np.zeros(num_feats)
        self.avg_weights = np.zeros(num_feats)
        self.avg_bias = 0

    def sgn(self, sample):
        dot_product = 0
        #print sample
        for (j,v) in sample:
            dot_product += v * self.weights[j]
        dot_product += self.bias
        if dot_product > 0:
            return 1
        else:
            return -1

    def avg_sgn(self, sample):
        dot_product = 0
        for (j, v) in sample:
            dot_product += v * (self.weights[j] - self.avg_weights[i]/c)
        dot_product += self.bias - self.avg_bias/c
        dot_product > 0:
            return 1
        else:
            return -1
        
    def update_weights(self, x, y, c):
        # calculate bias 
        b = self.learn_rate * y
        self.bias += b
        self.avg_bias += b * c
        for (j,v) in x:
            w = self.learn_rate * y * v
            self.weights[j] += w
            self.avg_weights += w * c

    def fit(self, samples, labels, n_feats):
        # init weights for all features
        self.init_weights(n_feats)
        c = 1.0

        for i in xrange(self.max_iter):
            num_mistakes = 0

            for j in xrange(len(labels)):
                y_pred = self.sgn(samples[j])

                if y_pred != labels[j]:
                    num_mistakes += 1
                    self.update_weights(samples[j], labels[j])
                c += 1

            print i, " - # mistakes", num_mistakes

            if num_mistakes == 0:
                return
        return

    def predict(self, samples, avg=1):
        if avg:
            return [self.avg_sgn for s in samples]
        else:
            return [self.sgn(s) for s in samples]
