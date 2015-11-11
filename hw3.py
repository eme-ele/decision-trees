import optparse
import csv

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from feature_extractor import ngrams_fe
from subgradient_descent import subgradient_descent

def parse_and_validate():
    ## parse
    parser = optparse.OptionParser()
    parser.add_option('-i', dest ='max_iterations', type='int',
                      help='(int) max number of iterations. must be > 0', default=10)
    parser.add_option('-r', dest='regularization', type='string',
                      help='(string) l1 or l2 regularization', default='l1')
    parser.add_option('-s', dest='step_size', type='float',
                      help='(float) step size taken by GD. must be > 0 and <= 1', default=0.1)
    parser.add_option('-l', dest='lmbd', type='float',
                      help='(float) lambda argument. must be > 0 and <= 1', default=0.1)
    parser.add_option('-f', dest='feature_set', type='int', default=1,
                      help='(int) 1: unigrams, 2: bigrams, 3: unigrams + bigrams')

    (opts, args) = parser.parse_args()

    ## validate
    if (opts.max_iterations < 0) or\
       (opts.regularization not in ['l1', 'l2']) or\
       (not (opts.step_size > 0 and opts.step_size <= 1)) or\
       (not (opts.lmbd > 0 and opts.lmbd <= 1)) or\
       (opts.feature_set not in [1,2,3]):
           print "\ninvalid argument"
           parser.print_help()
           exit(-1)

    return opts

## Gradient Descent Algorithm
def GD(max_iterations, regularization, step_size, lmbd, feature_set,
       num_feats, samples, labels):
    sgd = subgradient_descent(max_iterations, regularization, step_size, lmbd)
    sgd.fit(samples, labels, num_feats)
    return sgd

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

## preprocessing
def preprocess(samples):
    samples = [s.lower() for s in samples]
    #samples = [s.split() for s in samples]
    return samples

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    opts = parse_and_validate()
    #print opts.max_iterations, opts.regularization, opts.step_size, opts.lmbd, opts.feature_set

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    train_samples, train_labels = parse_file("data/hw3/train.csv")
    test_samples, test_labels = parse_file("data/hw3/test.csv")

    print "num train_samples", len(train_labels),
    print "\tpos:", train_labels.count(1), "neg:", train_labels.count(-1)
    print "num test_samples", len(test_labels)
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
    print "extracting features for complete data (train/test)..."
    train_features = fe.extract(train_samples)
    test_features = fe.extract(test_samples)

    print "training classifier..."
    classifier = GD(opts.max_iterations, opts.regularization, opts.step_size,
                    opts.lmbd, opts.feature_set, fe.n_feats, train_features, train_labels)
    results = classifier.predict(test_features)

    print classification_report(test_labels, results)
    print accuracy_score(test_labels, results)
    print confusion_matrix(test_labels, results)



if __name__ == '__main__':
    main()
