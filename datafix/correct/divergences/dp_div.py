__author__ = "Ming Tu"
"""Calculating DP divergence using minimum spanning tree"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree


def dp_div(X1, X2):
    """
    Calculating the DP divergence between data set X1 and X2.
    :param X1: N by D matrix
    :param X2: N by D matrix
    :return: a floating number represent the DP divergence
    """

    m = X1.shape[0]
    n = X2.shape[0]
    dimX = X1.shape[1]
    dimY = X2.shape[1]
    N = m + n

    if m != n:
        print(
            "CAUTION: The data sets have different numbers of samples: %6d, %6d\n"
            % (m, n)
        )
    if (dimY > n) or (dimX > m):
        print(
            "CAUTION: Are you sure the sampes are in rows and the features are in columns?"
        )
    if dimX != dimY:
        print("Error: The data sets have different dimensionalities!")
        exit()

    # Options
    options = {}
    options["nTrees"] = 1
    options["pNorm"] = 2

    # Combine two datasets
    dataX1X2 = np.concatenate((X1, X2))
    # Maintain class information
    nodeX1 = -1 * np.ones([m, 1])
    nodeX2 = np.ones([n, 1])
    nodeX1X2 = np.concatenate((nodeX1, nodeX2))

    # Generate symmetric matrix of internode weights
    weightsOfEdges = squareform(pdist(dataX1X2, "minkowski", p=options["pNorm"]))
    # Generate a "large" weight, used to replace weights of edges already in
    # listOfEdges as we step through the nTree MSTs
    maxWeight = 10 * np.amax(weightsOfEdges)

    # Determine the list of edges in the nTrees MSTs
    listOfEdges = ([], [])
    currentMST = 0
    while currentMST < options["nTrees"]:
        # Get current MST
        MSTtree = minimum_spanning_tree(np.triu(weightsOfEdges))
        nonzeroindex = MSTtree.nonzero()

        # Knock out these edges from future consideration for orthogonal MSTs
        for idx1, idx2 in zip(nonzeroindex[0], nonzeroindex[1]):
            weightsOfEdges[idx1][idx2] = maxWeight
            weightsOfEdges[idx2][idx1] = maxWeight
        listOfEdges = (
            listOfEdges[0] + nonzeroindex[0].tolist(),
            listOfEdges[1] + nonzeroindex[1].tolist(),
        )
        currentMST = currentMST + 1

    nodes_dim1 = [nodeX1X2[index].tolist() for index in listOfEdges[0]]
    nodes_dim2 = [nodeX1X2[index].tolist() for index in listOfEdges[1]]
    S = len([i for i in range(len(nodes_dim1)) if nodes_dim1[i] != nodes_dim2[i]])

    dp = float(1) - float(2 * (S - options["nTrees"]) / (N * options["nTrees"]))

    if dp < 0:
        return 0
    elif dp > 1:
        return 1
    else:
        return dp


# Test code
if __name__ == "__main__":
    X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
    X2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 1000)
    dp_div_value = dp_div(X1, X2)
    plt.scatter(X1[:, 0], X1[:, 1], color="red")
    plt.scatter(X2[:, 0], X2[:, 1], color="blue")
    plt.show()
    print(dp_div_value)
