import numpy as np
import sys

class perceptron(object):

    def __init__(self, max_iter, learn_rate):
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.weights = np.array([])
        self.bias = 0 #bias
        self.avg_weights = np.array([])
        self.avg_bias = np.array([])
        self.c = 1.0

    # weights should be set to random small values
    def init_weights(self, num_feats):
        self.bias = 0
        self.weights = np.zeros(num_feats)
        self.avg_weights = np.zeros(num_feats)
        self.avg_bias = 0

    def sgn(self, sample):
        dot_product = 0
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
            dot_product += v * (self.weights[j] - self.avg_weights[j]/self.c)
        dot_product += self.bias - self.avg_bias/self.c
        if dot_product > 0:
            return 1
        else:
            return -1

    def update_weights(self, x, y):
        # calculate bias
        b = self.learn_rate * y
        self.bias += b
        self.avg_bias += b * self.c
        for (j,v) in x:
            w = self.learn_rate * y * v
            self.weights[j] += w
            self.avg_weights[j] += w * self.c

    def fit(self, samples, labels, n_feats):
        # init weights for all features
        self.init_weights(n_feats)
        for i in xrange(self.max_iter):
            num_mistakes = 0

            for j in xrange(len(labels)):
                y_pred = self.sgn(samples[j])

                if y_pred != labels[j]:
                    num_mistakes += 1
                    self.update_weights(samples[j], labels[j])
                self.c += 1

            print i, " - # mistakes", num_mistakes

            if num_mistakes == 0:
                return
        return

    def predict_one(self, sample, avg=1):
        return self.avg_sgn(sample) if avg else self.sgn(sample)

    def predict(self, samples, avg=1):
        return [self.predict_one(s) for s in samples]

class kernel_perceptron(object):

    def __init__(self, max_iter, kernel='linear'):
        self.max_iter = max_iter
        if kernel == 'poly':
            self.kernel = self.poly_kernel
        elif kernel == 'gaussian':
            self.kernel = self.gaussian_kernel
        else:
            self.kernel = self.linear_kernel
        self.alphas = np.array([])
        self.bias = 0
        self.sv_samples = []
        self.sv_labels = []

    def init_alphas(self, num_samples):
        self.alphas = np.zeros(num_samples)
        self.bias = 0

    def dot_product(self, x1, x2):
        dp = 0.0
        for (i, v) in x1:
            for (j, w) in x2:
                if i == j:
                    dp += v*w
        return dp

    def linear_kernel(self, x1, x2):
        return self.dot_product(x1, x2)

    def poly_kernel(self, x1, x2, p):
        return (1 + self.dot_product(x1,x2))**p

    def blowup_space(self, samples):
        n = len(samples)
        K = np.zeros([n,n])
        total = n*n
        for i in xrange(n):
            for j in xrange(i, n):
                p = self.kernel(samples[i],samples[j])
                K[i,j] = p
                K[j,i] = p
        return np.array(K)

    def filter_samples(self, samples, labels):
        filt_X = []
        filt_y = []
        sv_index = self.alphas > 1e-5
        for i in xrange(len(samples)):
            if sv_index[i]:
                filt_x.append(samples[i])
                filt_y.append(labels[i])
        return filt_X, filt_y

    def fit(self, samples, labels):
        num_samples = len(labels)
        self.init_alphas(num_samples)

        K = self.blowup_space(samples)

        for i in xrange(self.max_iter):
            num_mistakes = 0
            for j in xrange(num_samples):
                y = labels[i]
                y_pred = np.sign(np.sum(K[:,i], self.alphas * y))

                if y_pred != y:
                    self.alpha[i] += y
                    num_mistakes += 1

            print i, " - # mistakes", num_mistakes

            if num_mistakes == 0:
                break

        ## support vector for classification
        ## memorize examples for which mistakes have been made
        ## use alpha[i] != 0
        self.sv_samples, self.sv_labels = self.filter_samples(samples, labels)

    def predict(self, samples):
        results = []
        for s in samples:
            dot_product = 0
            for a, sv_y, sv_x in zip(self.alphas, self.sv_labels, self.sv_samples):
                dot_product += a * sv_y * self.kernel(samples, sv_x)
            y_pred = 1 if dot_product >= 0 else -1
            results.append(y_pred)
        return results


