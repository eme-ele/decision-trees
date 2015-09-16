from collections import Counter
import numpy as np
import math
import copy


class Node:

    def __init__(self):
        self.attribute = None
        self.children = {}
        self.label = None
        self.parent = None
        self.sample_stats = {}
        self.depth = None

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

    def get_depth(self):
        return self.depth

    def set_depth(self, depth):
        self.depth = depth

    def prune(self):
        pruned_leaves = []
        visited = set([])
        visited.add(self)
        queue = []
        queue.append(self)

        ## BFS to find leaves on pruned node
        ## subtree
        while len(queue) > 0:
            current = queue.pop(0)
            for child in current.children.values():
                if child not in visited:
                    ## add to return
                    if child.is_leaf():
                        pruned_leaves.append(child)
                    visited.add(child)
                    queue.append(child)

        ## now prune the subtree
        self.children = {}

        return pruned_leaves, len(visited)-1

    def count_nodes(self):
        ## bfs to count nodes
        ## only done for debugging
        visited = set([])
        queue = []
        queue.append(self)
        visited.add(self)

        while len(queue) > 0:
            current = queue.pop(0)
            for child in current.children.values():
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        return len(visited)

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

    ## str representation for debugging
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


    def __init__(self, root_node=None, max_depth=9, max_split=10):
        self.root_node = root_node
        self.leaves = []
        self.att_thresholds = []
        self.max_depth = max_depth
        self.max_split = max_split

    def entropy(self, labels):
        stats = Counter(labels)
        total = len(labels)*1.0

        if total == 0.0:
            return 0.0

        values = [x/total for x in stats.values()]
        values = [x*math.log(x,2) if x!= 0.0 else 0.0 for x in values]
        e = -1.0 * sum(values)
        return e


    def attribute_entropy(self, samples, labels, att_index):
        att_vector = samples[:,att_index]
        att_values = set(att_vector)


        att_to_consider = sorted(att_values & self.att_thresholds[att_index])

        att_stats = {}
        greater_than = 0
        for att_value in att_to_consider:
            att_stats[att_value] = sum(x > greater_than and x <= att_value for x in att_vector)
            greater_than = att_value

        attribute_entropy = 0.0
        total = len(labels)*1.0

        # append labels to the vector for future filtering
        sub_matrix = np.column_stack((att_vector, labels))

        greater_than = 0
        for att_value in sorted(att_values & self.att_thresholds[att_index]):
            # obtain the subset for the corresponding threshold
            sub_labels = sub_matrix[(sub_matrix[:,0] > greater_than) & \
                                    (sub_matrix[:,0] <= att_value)]
            # obtain the corresponding labels
            sub_labels = sub_labels[:,1]
            # prob of occurrence of the value of an attribute * its entropy
            attribute_entropy += \
                     att_stats[att_value]/total * self.entropy(sub_labels)
            greater_than = att_value

        # entropy accumulated over all values for an attribute
        return attribute_entropy


    def information_gain(self, samples, labels, att_index):
        return self.entropy(labels) - \
               self.attribute_entropy(samples, labels, att_index)


    def choose_best(self, samples, labels, available_atts):
        num_attributes = samples.shape[1]
        gains = [self.information_gain(samples, labels, i)
                 for i in available_atts ]

        best_index = gains.index(max(gains))
        best_feat = available_atts[best_index]
        available_atts.pop(best_index)

        return best_feat


    def threshold_candidates(self, samples, labels, k):
        att_thresholds = []
        total = len(labels)*1.0
        for i in xrange(samples.shape[1]):
            att_vector = samples[:,i]
            att_stats = Counter(att_vector)
            att_gains = {}

            for value in att_stats:
                sub_matrix = np.column_stack((att_vector, labels))
                sub_labels = sub_matrix[sub_matrix[:,0] == value][:,1]
                attribute_entropy = \
                     att_stats[value]/total * self.entropy(sub_labels)

                att_gains[value] = self.entropy(labels) - attribute_entropy

            top_attributes = set([a for (a,g) in Counter(att_gains).most_common(k)])
            top_attributes.add(max(att_vector))
            att_thresholds.append(top_attributes)
        self.att_thresholds = att_thresholds

    def fit(self, samples, labels):

        self.threshold_candidates(samples, labels, self.max_split)

        available_atts = [i for i in xrange(samples.shape[1])]
        self.root_node = self.build_tree(samples, labels, available_atts, 0)


    def build_tree(self, samples, labels, available_atts, depth):

        samples = samples[:]
        labels = labels[:]
        available_atts = available_atts[:]

        node = Node()
        node.set_depth(depth)
        #print "depth", node.get_depth()

        # ran out of attributes in this branch
        # or I have reached my maximum depth limit
        # return majority class
        if (len(available_atts) == 0) or (node.get_depth() == self.max_depth):
            counts = Counter(labels)
            node.set_label(counts.most_common(1)[0][0])
            self.leaves.append(node)

        # if all labels are the same
        # return a leaf node with such class
        elif all(x == labels[0] for x in labels):
            node.set_label(labels[0])
            self.leaves.append(node)

        else:

            att_index = self.choose_best(samples, labels, available_atts)

            node.set_attribute(att_index)
            node.set_sample_stats(Counter(labels))
            att_values = set(samples[:,att_index])

            greater_than = 0
            has_children = False

            for value in sorted(self.att_thresholds[att_index]):
                matrix = np.column_stack((samples, labels))

                matrix = matrix[(matrix[:,att_index] > greater_than) & \
                                (matrix[:,att_index] <= value)]

                if len(matrix) == 0:
                    continue
                has_children = True

                # recursive call
                child = self.build_tree(matrix[:,:-1], matrix[:,-1], available_atts, depth+1)

                # set the parent child
                node.set_child(value, child)
                # set the child parent
                child.set_parent(node, value)
                # set the number of samples for each class
                sample_stats = Counter(matrix[:,-1])
                child.set_sample_stats(sample_stats)


                greater_than = value

            if not has_children:
                counts = Counter(labels)
                node.set_label(counts.most_common(1)[0][0])
                self.leaves.append(node)

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
                    att_thresholds = sorted(self.att_thresholds[current.get_attribute()])

                    greater_than = 0
                    chosen_threshold = 0
                    for t in att_thresholds:
                        if sample_attribute > greater_than and sample_attribute <= t:
                            chosen_threshold = t
                            break
                        greater_than = t


                    # child exists
                    if current.has_child(chosen_threshold):
                        current = current.get_child(chosen_threshold)
                    else:
                        # select the majority label for the current node
                        label = current.sample_stats.most_common(1)[0][0]

            labels.append(label)
        return labels

    def metrics(self, results, labels):
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


    def get_accuracy(self, samples, labels):
        results = self.predict(samples)
        (tp, fp, fn, tn) = self.metrics(results, labels)
        return float(tp+tn)/(tp+tn+fp+fn)


    # receives validation data
    def reduced_error_pruning(self, val_samples, val_labels):
        print "\nTrying reduced error pruning"
        accuracy = self.get_accuracy(val_samples, val_labels)
        print "Initial accuracy on validation set", accuracy
        monitor_leaves = self.leaves[:]

        print "Initial node count", self.root_node.count_nodes()

        while len(monitor_leaves) > 0:

            leaf = monitor_leaves[0]

            (node, value) = leaf.get_parent()

            temp_node = copy.deepcopy(node)
            temp_node.prune()

            if temp_node.is_leaf():
                temp_node.label = temp_node.sample_stats.most_common(1)[0][0]

            temp_root = temp_node.get_root()
            temp_tree = DecisionTree(temp_root)
            temp_tree.att_thresholds = self.att_thresholds

            temp_accuracy = temp_tree.get_accuracy(val_samples, val_labels)

            if temp_accuracy >= accuracy:
                accuracy = temp_accuracy
                # includes current leaf and its sibilings

                pruned_leaves, num_pruned = node.prune()

                print "\tprunned {0} nodes and increased accuracy to: {1}".format(num_pruned, accuracy)

                for pl in pruned_leaves:
                    monitor_leaves.remove(pl)
                    self.leaves.remove(pl)

                node.label = node.sample_stats.most_common(1)[0][0]
                self.leaves.append(node)
                monitor_leaves.append(node)

                root = node.get_root()
                print "\tactual node count:", root.count_nodes()

            else:
                monitor_leaves.remove(leaf)


        print "Final node count", self.root_node.count_nodes()

