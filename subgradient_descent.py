import random

class subgradient_descent(object):

    def __init__(self, max_iter, num_feats, lmbd, regularization, step_size):
        # parameters
        self.max_iter = max_iter
        self.lmbd = lmbd
        self.regularization = regularization
        self.step_size = step_size
        self.num_feats = num_feats
        # weights and gradients
        self.weights = np.zeros(num_feats)
        self.bias = 0.0
        self.gradient_w = np.zeros(num_feats)
        self.gradient_b = 0.0

    # condition for hinge_loss subgradient: y (w * x + b)
    # if y (w * x + b) > 1, subgradient will be 0
    # in any other case, should calculate
    def hinge_loss_sg_condition(x, y):
        # dot product between x and weights
        dot_product = 0.0
        for (j, v) in x:
            dot_product += v * self.weights[j]
        dot_product += self.bias

        return dot_product*y

    # update gradients with the subgradient
    # it is called only when subgradient != 0
    def update_gradients(x, y):
        sub_gradient = np.zeros(len())
        for (j, v) in x:
            sub_gradient_j = y * v
            self.gradient_w[j] += sub_gradient_j
        self.gradient_b += y

    def l2_update():
        for i in xrange(0, self.num_feats):
            self.gradient_w[i] -= self.lmbd * self.weights[i]

    def l1_update():
        for i in xrange(0, self.num_feats):
            self.gradient_w[i] -= self.lmbd

    def update_weights():
        for i in xrange(0, self.num_feats):
            self.weights[i] += self.step_size * self.gradient_w[i]
        self.bias += self.step * self.gradient_b

    def fit(self, samples, labels):
        num_samples = samples.shape[0]

        for i in xrange(self.max_iter):
            self.gradient_w = np.zeros(self.num_feats)
            self.gradient_b = 0.0

            for d in xrange(num_samples):
                sg_condition = self.hinge_loss_sg_condition(samples[d], labels[d])
                # if subgradient is not 0
                if sg_condition <= 1:
                    self.update_gradients()

            # add in regularization term
            if self.regularization == 'l2':
                self.l2_update()
            else:
                self.l1_update()

            # updates both weights and bias
            self.update_weights()

    def predict_one(self, sample):
        dot_product = 0.0
        for (j, v) in sample:
            dot_product += v * self.weights[j]
        dot+product += self.bias
        if dot_product > 0:
            return 1
        else:
            return -1

    def predict(self, samples):
        return [self.predict_one(s) for s in samples]
