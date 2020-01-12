"""Cluster elements through k-means."""
import sys
import random
from costfunctions import euclidean_sqr, euclidean, manhattan, pearson
from parsing import read_file


def kcluster(rows, distance=euclidean, k=4, max_retrys=10):
    """Perform k-means clustering."""
    current_retry = 0
    bestmatches_all_retrys = None
    bestdistance = None
    bestretry = None

    while current_retry != max_retrys:
        ranges = [(min([row[i] for row in rows]),
                   max([row[i] for row in rows])) for i in range(len(rows[0]))]

        clusters = [
            [random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                for i in range(len(rows[0]))] for _ in range(k)]

        lastmatches = None
        bestmatches = None

        for t in range(100):
            print("ITERATION NUM: %d, RETRY NUM: %d" %
                  (t + 1, current_retry + 1))
            bestmatches = [[]] * k
            total_distance = [0] * k

            # Find which centroid is the closest for each row
            for j in range(len(rows)):
                row = rows[j]
                bestmatch = 0
                for i in range(k):
                    best_distance = None
                    d = distance(clusters[i], row)
                    if d < distance(clusters[bestmatch], row):
                        bestmatch = i
                        best_distance = d

                bestmatches[bestmatch].append(j)
                if best_distance is not None:
                    total_distance[bestmatch] += best_distance
                else:
                    total_distance[bestmatch] += d

            # Equal matches on last iteration, solution found
            if bestmatches == lastmatches:
                break
            lastmatches = bestmatches

            # Move the centroids to the average of their members
            for i in range(k):
                avgs = [0.0] * len(rows[0])
                if len(bestmatches[i]) > 0:
                    for rowid in bestmatches[i]:
                        for m in range(len(rows[rowid])):
                            avgs[m] += rows[rowid][m]
                    for j in range(len(avgs)):
                        avgs[j] /= len(bestmatches[i])
                    clusters[i] = avgs

        if bestdistance is None or sum(total_distance) < bestdistance:
            bestmatches_all_retrys = bestmatches
            bestdistance = sum(total_distance)
            bestretry = current_retry
        current_retry += 1

    return bestmatches_all_retrys, bestdistance, bestretry


try:
    max_retrys = int(sys.argv[1])  # number of execution times
except Exception:
    print("Usage: python clusters.py <int: max_retrys>")
    sys.exit()

blognames, words, data = read_file('data/blogdata.txt', has_headers=True)
kclust, total_distance, best_retry = kcluster(data, k=4, max_retrys=max_retrys)

print("BEST CLUSTERING: " + str(kclust))
print("WITH A MINIMUM TOTAL DISTANCE OF " + str(total_distance))
print("THE BEST RETRY WAS NUMBER " + str(best_retry + 1))
