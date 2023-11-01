from absl import flags
from absl import logging
import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import time

from .utils import *

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "ghp_approx",
    None,
    "Number of nearest distances per node to consider",
    lower_bound=1,
)


def eval_from_matrix(train_features, train_labels):
    logging.log(
        logging.DEBUG, "Start computing distance matrix D with euclidian measure"
    )
    start = time.time()
    d = compute_distance_matrix_loo(train_features, "squared_l2")
    # d = np.sqrt(compute_distance_matrix_loo(train_features, "squared_l2"))

    if False:  # FLAGS.ghp_approx:
        indices = np.argpartition(d, FLAGS.ghp_approx, axis=1)[:, : FLAGS.ghp_approx]
        for row_i in range(d.shape[0]):
            mask = np.ones(d.shape[1], dtype=bool)
            mask[indices[row_i, :]] = False
            d[row_i, mask] = 0.0

    end = time.time()
    logging.log(logging.DEBUG, "D computed in {} seconds".format(end - start))

    logging.log(logging.DEBUG, "Start computing MST on matrix d")
    start = time.time()
    d = minimum_spanning_tree(d).tocoo()
    end = time.time()
    logging.log(logging.DEBUG, "MST computed in {} seconds".format(end - start))

    logging.log(logging.DEBUG, "Start computing estimator")
    start = time.time()

    classes, classes_counts = np.unique(train_labels, return_counts=True)

    num_train_samples = train_labels.size
    num_classes = len(classes)

    mapping = {}
    idx = 0
    for c in classes:
        mapping[c] = idx
        idx += 1

    deltas = []
    for i in range(num_classes - 1):
        deltas.append([0.0] * (num_classes - i - 1))

    # Claculate number of dichotomous edges
    for i in range(num_train_samples - 1):
        label_1 = mapping[train_labels[d.row[i]]]
        label_2 = mapping[train_labels[d.col[i]]]
        if label_1 == label_2:
            continue
        if label_1 > label_2:
            tmp = label_1
            label_1 = label_2
            label_2 = tmp
        deltas[label_1][label_2 - label_1 - 1] += 1

    # Devide the number of dichotomous edges by 2 * num_train_samples to get estimator of deltas
    deltas = [
        [item / (2.0 * num_train_samples) for item in sublist] for sublist in deltas
    ]

    # Sum up all the deltas
    delta_sum = sum([sum(sublist) for sublist in deltas])

    end = time.time()
    logging.log(logging.DEBUG, "Estimators computed in {} seconds".format(end - start))

    upper = 2.0 * delta_sum

    lower = ((num_classes - 1.0) / float(num_classes)) * (
        1.0
        - math.sqrt(
            max(0.0, 1.0 - ((2.0 * num_classes) / (num_classes - 1.0) * delta_sum))
        )
    )

    return {"default": [upper, lower]}
