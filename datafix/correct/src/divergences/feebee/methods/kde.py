from absl import flags
from absl import logging
import numpy as np
import math
from sklearn.neighbors import KernelDensity
import scipy.special
import time

from .utils import *

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "kde_knn_measure",
    "squared_l2",
    ["squared_l2", "cosine"],
    "Measure used for KNN distance matrix computation",
)
flags.DEFINE_multi_integer("kde_knn_k", "1", "Values for K in KNN", lower_bound=1)
flags.DEFINE_enum(
    "kde_kernel",
    "gaussian",
    ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"],
    "Kernel used for sk-learn kde method",
)
flags.DEFINE_float(
    "kde_bandwidth", 1.0, "Bandwidth for corresponding kernel", lower_bound=0.0
)

KEY_KNN_PATTERN = "measure={0}, k={1}"
KEY_KDE_PATTERN = "kernel={0}, bandwidth={1}"


def eval_from_matrix_knn_loo(train_features, train_labels):
    logging.log(
        logging.DEBUG,
        "Start computing LOO distance matrix with measure='{}'".format(
            FLAGS.kde_knn_measure
        ),
    )
    start = time.time()
    d = compute_distance_matrix_loo(train_features, FLAGS.kde_knn_measure)
    end = time.time()
    logging.log(logging.DEBUG, "D computed in {} seconds".format(end - start))

    results = {}
    num_test = train_features.shape[0]

    ks = sorted(set(FLAGS.kde_knn_k), reverse=True)
    logging.log(logging.DEBUG, "Start posterior estimate for ks={}".format(ks))
    start = time.time()
    for k in ks:
        if k == 1:
            err_up = 0.0
            err_low = 0.0
        else:
            # Get nearest k neighbors
            indices_up = np.argpartition(d, k - 1, axis=1)
            indices_low = np.argpartition(
                d, k - 2, axis=1
            )  # needed as the argpartiton does only partiton not sort the smallest k items!
            err_up = 0.0
            err_low = 0.0
            for i in range(num_test):
                labels = train_labels[indices_up[i, :k]]
                keys, counts = np.unique(labels, return_counts=True)
                maxcnt = float(counts.max())
                err_up += maxcnt / k

                labels = train_labels[indices_low[i, : k - 1]]
                keys, counts = np.unique(labels, return_counts=True)
                for index, value in enumerate(keys):
                    if train_labels[i] == value:
                        counts[index] += 1
                        break
                maxcnt = float(counts.max())
                err_low += maxcnt / k

                # Resubstitute does not give always a lower bound!
                if err_low < err_up:
                    tmp = err_up
                    err_up = err_low
                    err_low = err_up

            err_up = 1.0 - (err_up / num_test)
            err_low = 1.0 - (err_low / num_test)
            if err_low > err_up:
                tmp = err_low
                err_low = err_up
                err_up = tmp
        results[KEY_KNN_PATTERN.format(FLAGS.kde_knn_measure, k)] = [err_up, err_low]
    end = time.time()
    logging.log(
        logging.DEBUG, "Posterior estimates calcuated in {} seconds".format(end - start)
    )

    return results


def eval_from_matrix_kde(train_features, train_labels):
    kde_kernel = "gaussian"
    kde_bandwidth = "silverman"

    classes, counts = np.unique(train_labels, return_counts=True)
    logging.log(
        logging.DEBUG,
        "Start estimating the class likelyhood with args='{}, {}'".format(
            kde_kernel, kde_bandwidth
        ),
    )
    start = time.time()
    kdes = []
    fracs = []
    for i, c in enumerate(classes):
        # get only samples for that class
        indices = np.where(train_labels == c)[0]
        kde = KernelDensity(kernel=kde_kernel, bandwidth=kde_bandwidth).fit(
            train_features[indices, :]
        )
        kdes.append(kde)
        fracs.append(counts[i] / float(len(train_labels)))
    end = time.time()
    logging.log(
        logging.DEBUG, "Likelyhoods estimated in {} seconds".format(end - start)
    )

    results = {}

    logging.log(logging.DEBUG, "Looping through all the samples to get max Posterior")
    start = time.time()
    err = 1.0
    sum_test = 0.0
    lst = [
        scipy.special.expit(k.score_samples(train_features))
        for (j, k) in enumerate(kdes)
    ]
    p_x_y = np.array(lst)
    p_x_y = (p_x_y.T / np.sum(p_x_y, axis=1)).T * np.array(fracs).reshape(-1, 1)
    err = 1.0 - np.sum(np.max(p_x_y, axis=0))
    end = time.time()
    logging.log(
        logging.DEBUG,
        "Max posterior for all test samples calculated in {} seconds".format(
            end - start
        ),
    )

    results["default"] = [err, err]

    return results
