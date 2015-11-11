import random
import numpy as np

class subgradient_descent(object):

    def __init__(self, max_iter, regularization, step_size, lmbd):
        # parameters
        self.max_iter = max_iter
        self.regularization = regularization
        self.step_size = step_size
        self.lmbd = lmbd

    # condition for hinge_loss subgradient: y (w * x + b)
    # if y (w * x + b) > 1, subgradient will be 0
    # in any other case, should calculate
    def hinge_loss_term(self, x, y):
        dot_product = 0.0
        for (j, v) in x:
            dot_product += v * self.weights[j]
        dot_product += self.bias
        return dot_product*y

    def hinge_loss(self, x, y):
        term = self.hinge_loss_term(x, y)
        return max(0, 1 - term)

    def avg_loss(self, samples, labels):
        loss = 0.0
        for (x,y) in zip(samples, labels):
            loss += self.hinge_loss(x,y)
        return loss/len(labels)

    # update gradients with the subgradient
    # it is called only when subgradient != 0
    def update_gradients(self, x, y):
        for (j, v) in x:
            sub_gradient_j = y * v
            self.gradient_w[j] += sub_gradient_j
        self.gradient_b += y

    def l2_update(self):
        for i in xrange(0, self.num_feats):
            self.gradient_w[i] -= self.lmbd * self.weights[i]

    def l1_update(self):
        for i in xrange(0, self.num_feats):
            self.gradient_w[i] -= self.lmbd

    def update_weights(self):
        for i in xrange(0, self.num_feats):
            self.weights[i] += self.step_size * self.gradient_w[i]
        self.bias += self.step_size * self.gradient_b

    def fit(self, samples, labels, num_feats):
        num_samples = len(labels)
        #print "num_samples", num_samples
        #print labels
        #exit()
        self.num_feats = num_feats

        # weights
        self.weights = np.zeros(num_feats)
        self.bias = 0.0

        for i in xrange(self.max_iter):
            # gradients
            self.gradient_w = np.zeros(self.num_feats)
            self.gradient_b = 0.0

            for d in xrange(num_samples):
                x_d = samples[d]
                y_d = labels[d]
                # if subgradient is not 0
                if self.hinge_loss_term(x_d, y_d) <= 1:
                    self.update_gradients(x_d, y_d)

            # add in regularization term
            if self.regularization == 'l2':
                self.l2_update()
            else:
                self.l1_update()

            # updates both weights and bias
            self.update_weights()
            print "Iter", i, "Avg Loss", self.avg_loss(samples, labels)

    def predict_one(self, sample):
        dot_product = 0.0
        for (j, v) in sample:
            dot_product += v * self.weights[j]
        dot_product += self.bias
        if dot_product < 0:
            return -1
        else:
            return 1

    def predict(self, samples):
        return [self.predict_one(s) for s in samples]
