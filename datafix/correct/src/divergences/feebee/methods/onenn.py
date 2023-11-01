from absl import flags
from absl import logging
import numpy as np
import math
from sklearn.neighbors import KernelDensity
import time

from .utils import *
from .knn import _get_lowerbound

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "onenn_measure",
    "squared_l2",
    ["squared_l2", "cosine"],
    "Measure used for KNN distance matrix computation",
)
flags.DEFINE_multi_integer(
    "onenn_k", "1", "Values for K to estimate the 1NN error", lower_bound=1
)

KEY_PATTERN = "measure={0}, k={1}"


def eval_from_matrix_onenn(train_features, train_labels):
    logging.log(
        logging.DEBUG,
        "Start computing LOO distance matrix with measure='{}'".format(
            FLAGS.onenn_measure
        ),
    )
    start = time.time()
    d = compute_distance_matrix_loo(train_features, FLAGS.onenn_measure)
    end = time.time()
    logging.log(logging.DEBUG, "D computed in {} seconds".format(end - start))

    classes = len(np.unique(train_labels))

    results = {}
    num_test = train_features.shape[0]

    ks = sorted(set(FLAGS.onenn_k), reverse=True)
    logging.log(logging.DEBUG, "Start 1NN estimate using ks={}".format(ks))
    start = time.time()
    for k in ks:
        if k == 1:
            err = 0.0
        else:
            # Get nearest k-1 neighbors (resubstitution mode)
            indices = np.argpartition(d, k - 2, axis=1)
            err = 0.0
            for i in range(num_test):
                labels = train_labels[indices[i, : k - 1]]

                keys, counts = np.unique(labels, return_counts=True)
                for index, value in enumerate(keys):
                    if train_labels[i] == value:
                        counts[index] += 1
                        break

                for k_c in counts:
                    err += k_c * (k - k_c)

            err = float(err) / (num_test * (k * (k - 1)))
        results[KEY_PATTERN.format(FLAGS.onenn_measure, k)] = [
            err,
            _get_lowerbound(err, 1, classes),
        ]
    end = time.time()
    logging.log(
        logging.DEBUG, "1NN estimates calcuated in {} seconds".format(end - start)
    )

    return results
