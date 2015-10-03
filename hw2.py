import sys
import optparse
import csv
from feature_extractor import *

# template.py
# -------
# YOUR NAME HERE

##predict a single example
def predict_one(weights, input_snippet):
    pass
    return sign

##Perceptron
#-----------
def perceptron(maxIterations, featureSet):
    pass
    return weights


##Winnow
#-------
def winnow(maxIterations, featureSet):
    pass
    return weights


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
                      help='1: unigrams, 2: bigrams, 3: both')

    (opts, args) = parser.parse_args()
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
    print "num_samples", len(train_labels),
    print "positive:", train_labels.count(1), "negative:", train_labels.count(-1)

    print "preprocessing..."
    train_samples = preprocess(train_samples)

    fe = ngrams_fe(1)

    print "training..."
    feats = fe.train(train_samples, train_labels)

    print "extracting..."
    ret = fe.extract(train_samples)

    print len(ret)

    print ret


if __name__ == '__main__':
    main()
