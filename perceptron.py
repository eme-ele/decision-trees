import numpy as np

class Perceptron(object):

    def __init__(self, max_iter, learn_rate):
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.weights = np.array([])

    # weights should be set to random small values
    def init_weights(self, num_feats):
        self.weights = np.zeros(num_feats)

    def sgn(self, sample):
        x = sample[:]
        dot_product = np.dot(x, self.weights)
        if dot_product > 0:
            return 1
        else:
            return -1

    def update_weights(x, y):
        self.weights = self.weights + self.learn_rate * y * x
        #   self.weights[i] = self.weights[i] + \
        #                      self.learn_rate * y * x[i]

    def fit(self, samples, labels):
        # init weights for all features
        self.init_weights(samples.shape[1])

        ## add x0 = 1 for all samples
        np.insert(xs, 0, 1, axis=1)
        for i in xrange(self.max_iter):
            num_mistakes = 0

            for i in xrange(len(labels)):
                x = xs[i]
                y = labels[i]
                y_pred = self.sgn(x)

                if y_pred != y:
                    num_mistakes+=1
                    self.update_weights(x, y)

            print i, " - # mistakes", num_mistakes

            if num_mistakes == 0:
                return
        return

    def predict(self, sample):
        ## add x0 = 1 for all samples
        np.insert(xs, 0, 1, axis=1)
        return [sgn(x) for x in xs]
