# main.py
# -------
# YOUR NAME HERE

import sys
import optparse
import numpy as np
from decision_tree import DecisionTree
#from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# classify
#---------

def classify(decisionTree, example):

    print "\nClassifying with {0} nodes...".format(decisionTree.root_node.count_nodes())

    results = decisionTree.predict(example)
    return results

def metrics(results, labels):
    tp = 0; fp = 0; fn = 0; tn = 0
    for true_class, pred_class in zip(labels, results):
        if true_class == 1 and pred_class == 1:
            tp += 1
        elif true_class == 1 and pred_class == 0:
            fn += 1
        elif true_class == 0 and pred_class == 1:
            fp += 1
        elif true_class == 0 and pred_class == 0:
            tn += 1
    return tp, fp, fn, tn

def get_accuracy(results, labels):
    (tp, fp, fn, tn) = metrics(results, labels)
    return float(tp+tn)/(tp+tn+fp+fn)

def get_prec_recall(results, labels):
    (tp, fp, fn, tn) = metrics(results, labels)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    return precision, recall

def confusion_matrix(results, labels):
    (tp, fp, fn, tn) = metrics(results, labels)
    return np.array([[tp, fn],[fp, tn]])


def report_results(results, true_labels):
    print "\nAccuracy", get_accuracy(results, true_labels)
    (precision, recall) = get_prec_recall(results, true_labels)
    print "Precision", precision
    print "Recall", recall
    print "Confusion Matrix"
    print confusion_matrix(results, true_labels)


# learn
#-------
def learn(samples, labels, max_depth=9, max_split=10):
    learner = DecisionTree(max_depth=max_depth, max_split=max_split)
    learner.fit(samples, labels)
    return learner


# parse_file
# helper to parse data files
#-------
def parse_file(filename):
    with open(filename) as f:
        matrix = []
        for line in f:
            sample = line.strip().split()
            sample = map(int, sample)
            matrix.append(sample)
    matrix = np.array(matrix)
    samples = matrix[:,:-1]
    labels = matrix[:,-1]
    return samples, labels


def tune_depth(train_samples, train_labels, val_samples, val_labels):
    max_accuracy = 0.0
    best_depth = 0
    for i in range(0,10):
        learner = learn(train_samples, train_labels, max_depth=i)
        results = classify(learner, val_samples)
        acc_score = get_accuracy(results, val_labels)
        print "accuracy with depth", i, acc_score
        if acc_score > max_accuracy:
            best_depth = i
            max_accuracy = acc_score

    return best_depth


def tune_split(train_samples, train_labels, val_samples, val_labels):
    max_accuracy = 0.0
    best_split = 0
    for i in range(5,11):
        learner = learn(train_samples, train_labels, max_split=i)
        results = classify(learner, val_samples)
        acc_score = get_accuracy(results, val_labels)
        print "accuracy with split", i, acc_score
        if acc_score > max_accuracy:
            best_split = i
            max_accuracy = acc_score

    return best_split


# main
# I changed the input parsing method for a simpler library
# ----

def main():

    parser = optparse.OptionParser()
    parser.add_option('-p', dest='pruning', default=False, \
                      action='store_true', help='include pruning')
    parser.add_option('-t', dest='train_set', type='string',\
                      help='training data file')
    parser.add_option('-v', dest='val_set', type='string',\
                      help='validation data file')
    parser.add_option('-o', dest='test_set', type='string',\
                      help='testing data file')
    parser.add_option('-d', dest='max_depth', type='int', default=9)
    parser.add_option('-s', dest='max_split', type='int', default=10)
    parser.add_option('--td', dest='tune_depth', default=False, \
                      action='store_true', help='tune max_depth param')
    parser.add_option('--ts', dest='tune_split', default=False, \
                      action='store_true', help='tune max_split param')

    (opts, args) = parser.parse_args()
    mandatories = ['train_set', 'test_set', 'val_set']
    for m in mandatories:
        if not opts.__dict__[m]:
            print "mandatory option is missing\n"
            parser.print_help()
            exit(-1)


    '''arguments = validateInput(sys.argv)
    pruneFlag, valSetSize, maxDepth = arguments
    print pruneFlag, valSetSize, maxDepth'''

    # File parsing
    train_samples, train_labels = parse_file(opts.train_set)
    test_samples, test_labels = parse_file(opts.test_set)
    val_samples, val_labels = parse_file(opts.val_set)

    # Training
    learner = learn(train_samples, train_labels, opts.max_depth, opts.max_split)

    results = classify(learner, test_samples)
    report_results(results, test_labels)


    if opts.tune_depth:
        opts.max_depth = tune_depth(train_samples, train_labels, val_samples, val_labels)


    if opts.tune_split:
        opts.max_split = tune_split(train_samples, train_labels, val_samples, val_labels)

    if opts.tune_depth or opts.tune_split:

        print "\nTesting with adjusted max_depth: ", opts.max_depth, "and adjusted max_split: ", opts.max_split

    learner = learn(train_samples, train_labels, opts.max_depth, opts.max_split)
    results = classify(learner, test_samples)
    report_results(results, test_labels)


    learner.reduced_error_pruning(val_samples, val_labels)
    results = classify(learner, test_samples)
    report_results(results, test_labels)


if __name__ == '__main__':
    main()



