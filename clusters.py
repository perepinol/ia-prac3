import math
import sys
import random
from math import sqrt


def readfile(filename):
    with open(filename) as file_dat:
        lines = [line for line in file_dat]
        colnames = lines[0].strip().split('\t')[1:]
        rownames = []
        data = []
        for line in lines[1:]:
            p = line.strip().split('\t')
            rownames.append(p[0])
            data.append([float(x) for x in p[1:]])

    return rownames, colnames, data


def euclidean_sqr(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return result


def euclidean(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return sqrt(result)


def manhattan(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += abs((v1[i] - v2[i]))
    return result


def pearson(v1, v2):
    sum1 = sum(v1)
    sum2 = sum(v2)

    sum1Sq = sum([pow(v, 2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])

    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
    if den == 0:
        return 0
    return 1.0 - num / den


class bicluster:
    def __init__(self, vec, left=None, right=None, dist=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist


def hcluster(data, distance=euclidean):
    distances = {}  # stores the distances for efficiency
    currentclustid = -1  # all except the original items have a negative id

    # Clusters are initially just the rows
    clust = [bicluster(data[i], id=i) for i in range(len(data))]
    while len(clust) > 1:  # stop when there is only one cluster left
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        # calculate the average of the two clusters
        mergevec = [
            (clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0
            for i in range(len(clust[0].vec))]
        # create the new cluster
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]],
                               right=clust[lowestpair[1]],
                               dist=closest, id=currentclustid)

        # cluster ids that were not in the original set are negative
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]


def rotatematrix(data_matrix):
    return [list(elem) for elem in zip(*data_matrix)]


def kcluster(rows, distance=euclidean, k=4, max_retrys=10):
    current_retry = 0
    bestmatches_all_retrys = None
    bestdistance = None
    bestretry = None

    while current_retry != max_retrys:
        ranges = [(min([row[i] for row in rows]),
                   max([row[i] for row in rows])) for i in range(len(rows[0]))]

        clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                     for i in range(len(rows[0]))] for j in range(k)]

        lastmatches = None
        bestmatches = None

        for t in range(100):
            print("ITERATION NUM:" + str(t + 1) + ", RETRY NUM: " + str(current_retry + 1))
            bestmatches = [[] for i in range(k)]
            total_distance = [0 for i in range(k)]

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

blognames, words, data = readfile('data/blogdata.txt')
kclust, total_distance, best_retry = kcluster(data, k=4, max_retrys=max_retrys)

print("BEST CLUSTERING: " + str(kclust))
print("WITH A MINIMUM TOTAL DISTANCE OF " + str(total_distance))
print("THE BEST RETRY WAS NUMBER " + str(best_retry + 1))
