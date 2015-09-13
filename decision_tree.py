from collections import Counter
import numpy as np
import math

cuenta = 0

class Node:

    def __init__(self):
        self.attribute = None
        self.children = {}
        self.label = None

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




class DecisionTree:


    def __init__(self):
        self.root_node = None


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


    def fit(self, samples, labels):
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

        # if all labels are the same, return a leaf node
        elif all(x == labels[0] for x in labels):
            node = Node()
            node.set_label(labels[0])

        else:
            att_index = self.choose_best(samples, labels, available_atts)
            node = Node()
            node.set_attribute(att_index)
            att_values = set(samples[:,att_index])


            for value in att_values:
                matrix = np.column_stack((samples, labels))
                matrix = matrix[matrix[:,att_index] == value]

                child = self.build_tree(matrix[:,:-1], matrix[:,-1], available_atts)
                node.set_child(value, child)

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
                        label = -1

            labels.append(label)
        return labels

