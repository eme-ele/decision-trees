from collections import Counter
import numpy as np
import math
import copy
from sklearn.metrics import accuracy_score


class Node:

    def __init__(self):
        self.attribute = None
        self.children = {}
        self.label = None
        self.parent = None
        self.sample_stats = {}

    def is_leaf(self):
        return len(self.children) == 0

    def get_label(self):
        return self.label

    def set_label(self, label):
        self.label = label

    def set_attribute(self, attribute):
        self.attribute = attribute

    def get_attribute(self):
        return self.attribute

    def set_child(self, attribute_value, child):
        self.children[attribute_value] = child

    def has_child(self, attribute_value):
        return attribute_value in self.children

    def get_child(self, attribute_value):
        return self.children[attribute_value]

    def get_children(self):
        return self.children

    def prune_leaf(self, value):
        self.children.pop(value)

    def set_parent(self, node, value):
        self.parent = (node, value)

    def has_parent(self):
        return self.parent is not None

    def get_parent(self):
        return self.parent

    def set_sample_stats(self, sample_stats):
        self.sample_stats = sample_stats

    def get_sample_stats(self):
        return sample_stats

    def get_root(self):
        node = self
        while (node.parent is not None):
            node = node.parent[0]
        return node

    '''def __str__(self):
        string = "parent:"
        if self.parent is not None:
            string += "\n"
            for line in str(self.parent).split("\n"):
                string += "\t"+line+"\n"
        else:
            string += " None\n"

        string += "attribute: {0}\n".format(self.attribute)
        string += "children_values: {0}\n".format(self.children.keys())
        string += "label_counts: {0}\n".format(self.sample_stats)

        return string'''


class DecisionTree:


    def __init__(self, root_node=None):
        self.root_node = root_node
        self.leaves = []

    def entropy(self, labels):
        stats = Counter(labels)
        total = len(labels)*1.0

        if total == 0.0:
            return 0.0

        values = map(lambda x: x/total, stats.values())
        e = -1.0 * reduce(lambda x,y: x + ((y * math.log(y,2)) \
                                      if y != 0.0 else 0.0),\
                                      values)
        return e


    def attribute_entropy(self, samples, labels, att_index):
        att_vector = samples[:,att_index]
        att_stats = Counter(att_vector)
        attribute_entropy = 0.0
        total = len(labels)*1.0

        # append labels to the vector for future filtering
        sub_matrix = np.column_stack((att_vector, labels))
        for att_value in att_stats:
            # get the labels (second column) corresponding to the submatrix
            # of the att_value occurrence
            sub_labels = sub_matrix[sub_matrix[:,0] == att_value][:,1]
            # prob of occurrence of the value of an attribute * its entropy
            attribute_entropy += \
                     att_stats[att_value]/total * self.entropy(sub_labels)

        # entropy accumulated over all values for an attribute
        return attribute_entropy


    def information_gain(self, samples, labels, att_index):
        return self.entropy(labels) - \
               self.attribute_entropy(samples, labels, att_index)


    def choose_best(self, samples, labels, available_atts):
        num_attributes = samples.shape[1]
        gains = [self.information_gain(samples, labels, i)
                 for i in available_atts]
        best_index = gains.index(max(gains))
        best_feat = available_atts[best_index]
        available_atts.pop(best_index)
        return best_feat


    def threshold_candidates(self, samples, labels):
        att_thresholds = []
        total = len(labels)*1.0
        for i in xrange(samples.shape[1]):
            att_vector = samples[:,1]
            att_stats = Counter(att_vector)
            att_gains = []

            for value in att_stats:
                sub_matrix = np.column_stack((att_vector, labels))
                sub_labels = sub_matrix[sub_matrix[:,0] == value][:,1]
                attribute_entropy = \
                     att_stats[value]/total * self.entropy(sub_labels)

                att_gains.append(self.entropy(labels) - attribute_entropy)
            print att_gain
            print np.argsort(att_gains)[::-1][:5]
            exit(-1)

    def fit(self, samples, labels):
        #self.threshold_candidates(samples, labels)

        available_atts = [i for i in xrange(samples.shape[1])]
        self.root_node = self.build_tree(samples, labels, available_atts)


    def build_tree(self, samples, labels, available_atts):
        samples = samples[:]
        labels = labels[:]
        available_atts = available_atts[:]

        # ran out of attributes in this branch
        if len(available_atts) == 0:
            counts = Counter(labels)
            node = Node()
            node.set_label(counts.most_common(1)[0][0])
            self.leaves.append(node)

        # if all labels are the same, return a leaf node
        elif all(x == labels[0] for x in labels):
            node = Node()
            node.set_label(labels[0])
            self.leaves.append(node)

        else:
            att_index = self.choose_best(samples, labels, available_atts)
            node = Node()
            node.set_attribute(att_index)
            node.set_sample_stats(Counter(labels))
            att_values = set(samples[:,att_index])

            for value in att_values:
                matrix = np.column_stack((samples, labels))
                matrix = matrix[matrix[:,att_index] == value]

                # recursive call
                child = self.build_tree(matrix[:,:-1], matrix[:,-1], available_atts)

                # set the parent child
                node.set_child(value, child)
                # set the child parent
                child.set_parent(node, value)
                # set the number of samples for each class
                sample_stats = Counter(matrix[:,-1])
                child.set_sample_stats(sample_stats)

        return node


    def predict(self, samples):
        labels = []

        # a decision tree hasnt been trained
        if not self.root_node:
            return labels

        for s in samples:
            current = self.root_node
            label = None

            while label is None:
                # reached a leaf, return label
                if current.is_leaf():
                    label = current.get_label()
                else:
                    # go to child
                    sample_attribute = s[current.get_attribute()]
                    if current.has_child(sample_attribute):
                        current = current.get_child(sample_attribute)
                    else:
                        # select the majority label for the current node
                        label = current.sample_stats.most_common(1)[0][0]

            labels.append(label)
        return labels

    def get_accuracy(self, samples, labels):
        results = self.predict(samples)
        return accuracy_score(labels, results)

    # receives validation data
    def reduced_error_pruning(self, val_samples, val_labels):
        print "trying reduced error pruning"
        accuracy = self.get_accuracy(val_samples, val_labels)
        print "initial accuracy on validation set", accuracy
        monitor_leaves = self.leaves[:]

        while len(monitor_leaves) > 0:

            ## leaf and parent to work on
            leaf = monitor_leaves[0]
            (node, value) = leaf.get_parent()

            temp_node = copy.deepcopy(node)
            temp_node.prune_leaf(value)

            if temp_node.is_leaf():
                temp_node.label = temp_node.sample_stats.most_common(1)[0][0]

            temp_root = temp_node.get_root()
            temp_tree = DecisionTree(temp_root)
            temp_accuracy = temp_tree.get_accuracy(val_samples, val_labels)

            if temp_accuracy >= accuracy:
                accuracy = temp_accuracy
                print "\tprunned and increased accuracy to: ", accuracy
                # includes current leaf and its sibilings
                self.leaves.remove(leaf)
                monitor_leaves.remove(leaf)

                node.prune_leaf(value)

                if node.is_leaf():
                    node.label = node.sample_stats.most_common(1)[0][0]
                    self.leaves.append(node)
                    monitor_leaves.append(node)

            else:
                monitor_leaves.remove(leaf)

            del temp_node
