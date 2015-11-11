import numpy as np

def metrics(results, labels):
    tp = 0; fp = 0; fn = 0; tn = 0
    for true_class, pred_class in zip(labels, results):
        if true_class == 1 and pred_class == 1:
            tp += 1
        elif true_class == 1 and pred_class == -1:
            fn += 1
        elif true_class == -1 and pred_class == 1:
            fp += 1
        elif true_class == -1 and pred_class == -1:
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
