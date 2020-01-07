"""Build and classify elements using a decision tree."""
from costfunctions import calculate_probabilities, gini_impurity, entropy
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


def test_performance(dataset, training_ratio, scoref, beta, rounds=10000):
    """Test the decision tree's performance."""
    max_prob_success, success, failure = 0, 0, 0
    for _ in range(rounds):
        # Split dataset and train decision tree
        training, test = get_training_and_test_sets(dataset, training_ratio)
        tree = Node(training)
        tree.build_tree(scoref, beta)

        # Test decision tree
        for elem in test:
            probs = list(tree.classify(elem[:-1]).items())
            probs.sort(key=lambda x: x[1])

            # If expected class is not in probs, classification failed
            if len(probs) == 0 or elem[-1] not in map(lambda x: x[0], probs):
                failure += 1

            # If expected class is first, it is a top match
            elif elem[-1] == probs[0][0]:
                max_prob_success += 1

            # If expected class is not first, it is simple match
            else:
                success += 1

        for elem in test:
            probs = tree.classify(elem[:-1])

    # Output results
    tested = len(test) * rounds

    return {
        'function': str(scoref).split()[1],
        'beta': beta,
        'rounds': rounds,
        'set_size': len(dataset),
        'training_set_size': len(training),
        'test_set_size': len(test),
        'top_matches': 0 if tested == 0 else float(max_prob_success) / tested,
        'other_matches': 0 if tested == 0 else float(success) / tested,
        'failed_matches': 0 if tested == 0 else float(failure) / tested
    }


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
        self.probs = None

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

    def prune(self, scoref, threshold):
        """
        Prune the tree following a bottom-up strategy.

        Return the number of prunes made.
        """
        if not self._is_built():
            raise Exception("Decision tree is not built")

        # Base case: node is a leaf (otherwise, node always has two children)
        if self._is_leaf():
            return 0

        # Run prune on children
        pruned = self.tc.prune(scoref, threshold) + \
                 self.fc.prune(scoref, threshold)

        # If both children are leaves, see if they can be joined
        if self.tc._is_leaf() and self.fc._is_leaf():
            # Get impurity for the three nodes
            imp, tc_imp, fc_imp = map(scoref,
                map(lambda n: n.dataset, [self, self.tc, self.fc])
            )
            # If change in impurity is below threshold, remove leaves
            if imp - tc_imp - fc_imp < threshold:
                self.tc, self.fc = None, None
                self.probs = calculate_probabilities(self.dataset)
                return pruned + 1

        return pruned

    def classify(self, object):
        """Return the classification probability for a given object."""
        if not self._is_built():
            raise Exception("Decision tree is not built")

        if self._is_leaf():  # Base case
            return self.probs

        if Node._fulfills_criteria(object[self.col], self.value):
            return self.tc.classify(object)

        return self.fc.classify(object)

    def _find_children(self, scoref, beta):
        """Divide a node and assign the partitions as children if necessary."""
        if len(self.dataset) == 0:  # If dataset is empty, this is a leaf
            self.probs = {}
            return

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

    def _is_leaf(self):
        """Return True if the node is a leaf (has probs), False otherwise."""
        return self.probs is not None

    def _is_built(self):
        """Return True if the decision tree has been built, False otherwise."""
        return self.tc is not None or \
            self.fc is not None or \
            self._is_leaf()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Use: python %s <input file>" % sys.argv[0])
        sys.exit(0)

    input = parsing.read_file(sys.argv[1])
    tree = Node(input)

    print("Test 1: change in performance for different betas")
    print("| Beta | Top matches | Other matches | Failures |")
    best_beta, less_failure = None, None
    for beta in range(101):
        res = test_performance(input, 0.5, gini_impurity, beta / 100.0)
        print("| %3.2f | %10.2f%% | %12.2f%% | %7.2f%% |" % (
            res['beta'],
            res['top_matches'] * 100,
            res['other_matches'] * 100,
            res['failed_matches'] * 100
        ))
        if less_failure is None or res['failed_matches'] < less_failure:
            best_beta, less_failure = res['beta'], res['failed_matches']

    print("\nTest 2: change in performance for different training set size" +
          "(beta=%.2f)" % best_beta)
    print("| Training size | Test size | Top matches | Other matches | " +
          "Failures |")
    for set_size in range(11):
        res = test_performance(input, set_size / 10.0, gini_impurity, best_beta)
        print("| %13d | %9d | %10.2f%% | %12.2f%% | %7.2f%% |" % (
            res['training_set_size'],
            res['test_set_size'],
            res['top_matches'] * 100,
            res['other_matches'] * 100,
            res['failed_matches'] * 100
        ))
