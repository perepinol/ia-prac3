"""Build and classify elements using a decision tree."""
from costfunctions import calculate_probabilities, gini, entropy
import parsing


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

    def build_tree(self, scoref, beta=0):
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

    def build_tree_ite(self, scoref, beta=0):
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
    tree = Node(input)
    tree.build_tree(gini, 0.1)
    from printtree import printtree
    printtree(tree)
