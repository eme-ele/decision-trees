import sys
import optparse
import csv
from feature_extractor import *
import numpy as np
from perceptron import perceptron
from winnow import winnow

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# template.py
# -------
# YOUR NAME HERE

##predict a single example
def predict_one(weights, input_snippet):
    pass
    return sign

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
    if (opts.algorithm not in [1,2]) or \
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
    print "num train_samples", len(train_labels),
    print "\tpos:", train_labels.count(1), "neg:", train_labels.count(-1)
    print "num test samples", len(test_labels)
    print "\tpos:", test_labels.count(1), "neg:", test_labels.count(-1)

    print "preprocessing..."
    train_samples = preprocess(train_samples)
    test_samples = preprocess(test_samples)

    if opts.feature_set == 1:
        fe = ngrams_fe(1)
    elif opts.feature_set == 2:
        fe = ngrams_fe(2)
    elif opts.feature_set == 3:
        fe = ngrams_fe(1,2)

    print "training feature extractor..."
    fe.train(train_samples, train_labels)
    print "extracting features..."
    train_features = fe.extract(train_samples)

    print "training classifier..."
    if opts.algorithm == 1:
        classifier = perceptron(opts.max_iterations, 0.1)
    else:
        classifier = winnow(opts.max_iterations, 2)

    classifier.fit(train_features, train_labels, fe.n_feats)
    test_features = fe.extract(test_samples)
    results = classifier.predict(test_features)

    print classification_report(test_labels, results)
    print accuracy_score(test_labels, results)
    print confusion_matrix(test_labels, results)

if __name__ == '__main__':
    main()
