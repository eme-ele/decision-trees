# main.py
# -------
# YOUR NAME HERE

import sys
import optparse
import numpy as np
from decision_tree import DecisionTree
from sklearn.metrics import classification_report, accuracy_score


##Classify
#---------

def classify(decisionTree, example):
    results = decisionTree.predict(example)
    return results


##Learn
#-------
def learn(samples, labels):
    learner = DecisionTree()
    learner.fit(samples, labels)
    return learner


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

# main
# ----

def main():

    parser = optparse.OptionParser()
    parser.add_option('-p', dest='pruning', default=False, \
                      action='store_true', help='include pruning')
    parser.add_option('-t', dest='train_set', type='string',\
                      help='training data file')
    parser.add_option('-o', dest='test_set', type='string',\
                      help='testing data file')
    parser.add_option('-d', dest='max_depth', type='int')

    (opts, args) = parser.parse_args()
    mandatories = ['train_set', 'test_set']
    for m in mandatories:
        if not opts.__dict__[m]:
            print "mandatory option is missing\n"
            parser.print_help()
            exit(-1)


    '''arguments = validateInput(sys.argv)
    pruneFlag, valSetSize, maxDepth = arguments
    print pruneFlag, valSetSize, maxDepth'''

    train_samples, train_labels = parse_file(opts.train_set)
    learner = learn(train_samples, train_labels)

    test_samples, test_labels = parse_file(opts.test_set)
    results = classify(learner, test_samples)

    print(classification_report(test_labels, results))
    print(accuracy_score(test_labels, results))


if __name__ == '__main__':
    main()



