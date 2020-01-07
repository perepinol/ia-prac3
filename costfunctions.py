"""Contains cost functions for decision trees and K-means clustering."""


def calculate_probabilities(dataset):
    """Return the chances of an item belonging to each class in the set."""
    count = {}
    # Count class occurrences
    for element in dataset:
        data_class = element[-1]
        if data_class in count.keys():
            count[data_class] += 1
        else:
            count[data_class] = 1

    # Calculate probabilities
    for key, value in count.items():
        count[key] = float(value) / len(dataset)
    return count


def gini(dataset):
    """Calculate the Gini index for the given dataset."""
    from functools import reduce
    probs = calculate_probabilities(dataset)
    return 1 - reduce(lambda x, y: x + y, [p ** 2 for p in probs.values()], 0)


def entropy(dataset):
    """Calculate the entropy for a given dataset."""
    from math import log
    from functools import reduce
    probs = calculate_probabilities(dataset)
    entropies = [p * log(p, 2) for p in probs.values()]
    return -1 * reduce(lambda x, y: x + y, entropies, 0)


def euclidean_sqr(p1, p2):
    """Calculate the squared euc. distance between two n-dimensional points."""
    if len(p1) != len(p2):
        return -1
    return sum(
        map(
            lambda pair: (pair[0] - pair[1]) ** 2,
            zip(p1, p2)
        )
    )


def euclidean(p1, p2):
    """Calculate the euclidean distance between two n-dimensional points."""
    squared = euclidean_sqr(p1, p2)
    return squared if squared < 0 else float(squared) ** (1/2)


def manhattan(p1, p2):
    """Calculate the manhattan distance between two n-dimensional points."""
    if len(p1) != len(p2):
        return -1
    return sum(
        map(
            lambda pair: abs(pair[0] - pair[1]),
            zip(p1, p2)
        )
    )


def pearson(v1, v2):
    """Calculate the Pearson distance between two n-dimensional points."""
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of the squares
    sum1Sq = sum([pow(v, 2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])
    # Sum of the products
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])
    # Calculate r (Pearson score)
    num = pSum - (sum1 * sum2 / len(v1))
    den = (
        (sum1Sq - pow(sum1, 2) / len(v1)) *
        (sum2Sq - pow(sum2, 2) / len(v1))
    ) ** (1/2)
    if den == 0:
        return 0
    return 1.0-num/den
