"""Build and classify elements using a decision tree."""
from costfunctions import calculate_probabilities, gini, entropy
import parsing


def get_training_and_test_sets(dataset, training_ratio):
    """
    Split a dataset randomly in two.

    The first set will have training_ratio elements comapred to the full
    dataset.
    """
    import random
    test_set = dataset[:]
    training_set = []
    for _ in range(int(training_ratio * len(dataset))):
        training_set.append(test_set.pop(random.randrange(len(test_set))))
    return training_set, test_set


def test_performance(dataset, training_ratio, scoref, beta, rounds=100):
    """Test the decision tree's performance."""
    max_prob_success, success, failure = 0, 0, 0
    for _ in range(rounds):
        # Split dataset and train decision tree
        training, test = get_training_and_test_sets(dataset, training_ratio)
        tree = Node(training)
        tree.build_tree(gini, beta)
        from printtree import printtree
        #printtree(tree)
        # Test decision tree
        for elem in test:
            probs = list(tree.classify(elem[:-1]).items())
            probs.sort(key=lambda x: x[1])
            if elem[-1] == probs[0][0]:
                max_prob_success += 1
            elif elem[-1] in map(lambda x: x[0], probs):
                success += 1
            else:
                failure += 1

        for elem in test:
            probs = tree.classify(elem[:-1])

    # Output results
    tested = len(test) * rounds
    print("********* Test results *********")
    print("Function: %s" % str(scoref).split()[1])
    print("Beta: %.2f" % beta)
    print("Test rounds: %d" % rounds)
    print("Test set size: %d" % len(test))
    print("Element matched maximum probability: %d (%.2f%%)" %
          (max_prob_success, float(max_prob_success) / tested * 100))
    print("Element matched non-maximum probability: %d (%.2f%%)" %
          (success, float(success) / tested * 100))
    print("Element did not match: %d (%.2f%%)" %
          (failure, float(failure) / tested * 100))
    print("********************************")


class Node():
    """Describes a node in a decision tree."""

    def __init__(self, dataset=[], col=-1, value=None, tc=None, fc=None):
        """
        Node constructor.

        Stores the input dataset in the node, the division criteria (column
        and value), the node where the criteria is fulfilled and the one
        where it is not (if it is not leaf) and the probability of each
        class (if it is leaf).
        """
        self.dataset = dataset
        self.col = col
        self.value = value
        self.tc = None
        self.fc = None
        self.probs = {}

    def build_tree(self, scoref, beta):
        """
        Build the decision tree from this node downwards.

        Return the number of leaves.
        """
        self._find_children(scoref, beta)
        count = 0
        if self.tc is not None:
            count += self.tc.build_tree(scoref, beta)
        if self.fc is not None:
            count += self.fc.build_tree(scoref, beta)
        return 1 if count == 0 else count

    def build_tree_ite(self, scoref, beta):
        """
        Build the decision tree from this node downwards iteratively (BFS).

        Return the number of leaves.
        """
        pending_nodes = [self]
        count = 0
        while len(pending_nodes) > 0:
            node = pending_nodes.pop(0)
            node._find_children(scoref, beta)
            if node.tc is None and node.fc is None:
                count += 1
            if node.tc is not None:
                pending_nodes.append(node.tc)
            if node.fc is not None:
                pending_nodes.append(node.fc)
        return count

    def classify(self, object):
        """Return the classification probability for a given object."""
        # No need to check both children, if one is None, both are
        if self.tc is None and self.probs is None:
            raise Exception("Decision tree is not built")
        if len(self.probs.keys()) != 0:  # Base case
            return self.probs

        if Node._fulfills_criteria(object[self.col], self.value):
            return self.tc.classify(object)

        return self.fc.classify(object)

    def _find_children(self, scoref, beta):
        """Divide a node and assign the partitions as children if necessary."""
        curr_impurity = scoref(self.dataset)
        best = (beta, None, None, None, None)
        for col in range(len(self.dataset[0]) - 1):
            for value in set(map(lambda row: row[col], self.dataset)):
                s1, s2 = self._divide_dataset(col, value)
                goodness = \
                    curr_impurity - \
                    scoref(s1) * len(s1) / len(self.dataset) - \
                    scoref(s2) * len(s2) / len(self.dataset)
                if goodness > best[0]:
                    best = (goodness, col, value, s1, s2)
        if best[0] > beta:  # Node should be divided, assign children
            self.col, self.value = best[1:3]
            self.tc, self.fc = Node(best[3]), Node(best[4])
        else:  # Node should not be divided, assign probabilities
            self.probs = calculate_probabilities(self.dataset)

    def _divide_dataset(self, col, value):
        """
        Divide a dataset in disjunct parts.

        The criteria is the value (categorical/numeric) applied to the column.
        """
        set1, set2 = [], []
        for row in self.dataset:
            if (Node._fulfills_criteria(row[col], value)):
                set1.append(row)
            else:
                set2.append(row)
        return set1, set2

    @staticmethod
    def _fulfills_criteria(tested, reference):
        """
        Assert if the tested element is valid compared to the reference.

        Validity is checked by smaller-than-or-equal when the values are
        numerical and equality otherwise.
        """
        # Find out if reference is numerical or not
        numerical = parsing.is_numerical(reference)

        # Check criteria
        return \
            (numerical and tested <= reference) or \
            (not numerical and tested == reference)

    def __eq__(self, other):
        """Return true if other is Node, attributes and children are same."""
        if type(other) != Node:
            return False
        return \
            self.dataset == other.dataset and \
            self.col == other.col and \
            self.value == other.value and \
            self.probs == other.probs and \
            self.tc == other.tc and \
            self.fc == other.fc


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Use: python treepredict.py <input file>")
        sys.exit(0)

    input = parsing.read_file(sys.argv[1])
    test_performance(input, 0.5, gini, 0.2)
    test_performance(input, 0.5, gini, 0.1)
    test_performance(input, 0.5, gini, 0)
    print("\n")
    test_performance(input, 0.8, gini, 0.2)
    test_performance(input, 0.5, gini, 0.2)
    test_performance(input, 0.3, gini, 0.2)
