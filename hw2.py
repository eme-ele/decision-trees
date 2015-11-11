import optparse
import csv
from feature_extractor import ngrams_fe
from perceptron import perceptron as pc
from perceptron import kernel_perceptron as kpc
from winnow import winnow as wnn

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from metrics import report_results, get_accuracy

# template.py
# -------
# YOUR NAME HERE

def perceptron(max_iterations, learning_rate, num_feats, samples, labels):
    classifier = pc(max_iterations, learning_rate)
    classifier.fit(samples, labels, num_feats)
    return classifier

def kernel_perceptron(max_iterations, learning_rate, samples, labels):
    classifier = kpc(max_iterations, 'linear')
    classifier.fit(samples, labels)
    return classifier

def winnow(max_iterations, alpha, num_feats, samples, labels):
    classifier = wnn(max_iterations, alpha)
    classifier.fit(samples, labels, num_feats)
    return classifier

##predict a single example
def predict_one(classifier, sample):
    result = classifier.predict_one(sample)
    return result

## preprocessing
def preprocess(samples):
    samples = [s.lower() for s in samples]
    #samples = [s.split() for s in samples]
    return samples

## parse files
def parse_file(filename):
    samples = []
    labels = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            samples.append(row[0])
            labels.append(1 if row[1] == '+' else -1)
    return samples, labels

def tune_learning_rate(max_iterations, train_samples, train_labels,
                       val_samples, val_labels, num_feats):
    rates = [0.01, 0.05, 0.07, 0.1, 0.5, 0.7, 1]
    best = 0
    max_accuracy = 0.0
    for lr in rates:
        classifier = pc(max_iterations, lr)
        classifier.fit(train_samples, train_labels, num_feats)
        results = classifier.predict(val_samples)
        curr_accuracy = get_accuracy(results, val_labels)
        if curr_accuracy > max_accuracy:
            best = lr
            max_accuracy = curr_accuracy
    return best

def tune_alpha(max_iterations, train_samples, train_labels,
               val_samples, val_labels, num_feats):
    alphas = [0.1, 0.5, 1, 1.5, 2]
    best = 0
    max_accuracy = 0.0
    for a in alphas:
        classifier = wnn(max_iterations, alpha)
        classifier.fit(train_samples, train_labels, num_feats)
        results = classifier.predict(val_samples)
        curr_accuracy = get_accuracy(results, val_labels)
        if curr_accuracy > max_accuracy:
            best = a
            max_accuracy = curr_accuracy
    return best

# main
# ----
# The main program loop
# You should modify this function to run your experiments

# I changed the input method to optparse for convenience
def main():

    parser = optparse.OptionParser()
    parser.add_option('-a', dest='algorithm', default=1,
                      help='1: perceptron, 2: winnow', type='int')
    parser.add_option('-i', dest='max_iterations', default=10,
                      help='max number of iterations', type='int')
    parser.add_option('-f', dest='feature_set', default=1,
                      help='1: unigrams, 2: bigrams, 3: both', type='int')
    (opts, args) = parser.parse_args()
    print opts.feature_set

    ## no argument is mandatory, they all have defaults
    ## validate input
    if (opts.algorithm not in [1,2,3]) or \
       (opts.max_iterations < 0) or \
       (opts.feature_set not in [1,2,3]):
        print "invalid argument"
        parser.print_help()
        return

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    train_samples, train_labels = parse_file("data/hw2/train.csv")
    test_samples, test_labels = parse_file("data/hw2/test.csv")
    val_samples, val_labels = parse_file("data/hw2/validation.csv")
    print "num train_samples", len(train_labels),
    print "\tpos:", train_labels.count(1), "neg:", train_labels.count(-1)
    print "num test samples", len(test_labels)
    print "\tpos:", test_labels.count(1), "neg:", test_labels.count(-1)
    print "num val samples", len(test_labels)
    print "\tpos:", val_labels.count(1), "neg:", val_labels.count(-1)

    print "preprocessing..."
    train_samples = preprocess(train_samples)
    test_samples = preprocess(test_samples)
    val_samples = preprocess(val_samples)

    if opts.feature_set == 1:
        fe = ngrams_fe(1)
    elif opts.feature_set == 2:
        fe = ngrams_fe(2)
    elif opts.feature_set == 3:
        fe = ngrams_fe(1,2)

    print "training feature extractor..."
    fe.train(train_samples, train_labels)
    print "extracting features for complete data (train/test)..."
    train_features = fe.extract(train_samples)
    test_features = fe.extract(test_samples)
    val_features = fe.extract(val_samples)

    print "training classifier..."
    ## if perceptron, test both standard and averaged perceptron
    if opts.algorithm == 1:
        #best_lr = tune_learning_rate(opts.max_iterations, train_features,
        #                             train_labels, val_features, val_labels, fe.n_feats)
        classifier = perceptron(opts.max_iterations, 0.1, fe.n_feats,
                                train_features, train_labels)
    elif opts.algorithm == 2:
        #best_alpha = tune_alpha(opts.max_iterations, train_features,
        #                        train_labels, val_features, val_labels, fe.n_feats)
        classifier = winnow(opts.max_iterations, 2, fe.n_feats,
                            train_features, train_labels)
    elif opts.algorithm == 3:
        classifier = kernel_perceptron(opts.max_iterations, 'linear',
                                       train_features, train_labels)
    else:
        return

    results = classifier.predict(test_features)
    pred_res = classifier.predict(train_features)

    print classification_report(test_labels, results)
    print accuracy_score(test_labels, results)
    print confusion_matrix(test_labels, results)

    print classification_report(train_labels, pred_res)
    print accuracy_score(train_labels, pred_res)

    report_results(results, test_labels)

if __name__ == '__main__':
    main()
