from collections import Counter
import numpy as np

class Node:

    def __init__(self):
        self.attribute = None
        self.children = {}
        self.label = None

    def is_leaf():
        return len(self.children) == 0

    def get_label():
        return self.label

    def set_label(label):
        self.label = label

    def set_attribute(attribute):
        self.attribute = attribute

    def get_attribute():
        return self.attribute

    def set_child(attribute_value, child):
        self.children[attribute_value] = child

    def has_child(attribute_value):
        return attribute_value in self.children

    def get_child(attribute_value):
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
        sub_matrix = np.column_stack(att_vector, labels)
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


    def choose_best(self, samples, labels):
        num_attributes = samples.shape[1]
        gains = [self.information_gain(self, samples, labels, i)
                 for i in range(num_attributes)]
        return gain.index(max(gains))


    def learn(self, samples, labels):
        self.root_node = self.build_tree(samples, labels)


    def build_tree(self, samples, labels):
        # if all labels are the same, return a leaf node
        if all(x == labels[0] for x in labels):
            node = Node()
            node.set_label(labels[0])
            return node

        # if there are no more attributes, return a leaf node
        num_attributes = samples.shape[1]
        if num_attributes == 0:
            stats = Counter(labels)
            label = stats.most_common(1)[0][0]
            node = Node()
            node.set_label(label)
            return node

        attr_index = self.choose_best(samples, labels)
        node = Node()
        node.set_attribute(attr_index)

        att_values = set(samples[:,att_index])

        for value in att_values:
            matrix = np.column_stack(samples, labels)
            matrix = matrix[matrix[:,attr_index] == value]
            samples = matrix[:,:-1]
            labels = matrix[:,-1]

            child = self.build_tree(samples, labels)
            node.set_child(value, child)

        return node


    def classify(self, samples):
        # a decision tree hasnt been trained
        if not self.root_node:
            return -1
        # output labels
        labels = []

        for s in sample:
            current = self.root_node
            label = None

            while not label:
                # reached a leaf, return label
                if current.is_leaf():
                    label = current.get_label()
                # go to child
                sample_attribute = s[current.get_attribute()]
                if current.has_child(attribute_value):
                    current = current.get_child(attribute_value)

            labels.append(label)
        return labels

