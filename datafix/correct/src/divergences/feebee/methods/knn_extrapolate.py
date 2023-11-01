from absl import app
from absl import flags
from absl import logging
import numpy as np
import math
from scipy.optimize import curve_fit
import time

from .utils import *
from .knn import _get_lowerbound, eval_from_matrices_split

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "knn_extrap_samples",
    10,
    "Split the trainset into this number of samples for extrapolation",
    lower_bound=1,
)
flags.DEFINE_integer(
    "knn_extrap_order",
    5,
    "Degree of differentiability. Or two more than the degree of the polinomial.",
    lower_bound=3,
)

KEY_PATTERN = "measure=({0}, k={1}"


def extrapolate(x, y, classes, dimension):
    def paramed_curve(x, *params):
        bs = params[:-1]
        s = params[-1]
        for i in range(0, FLAGS.knn_extrap_order - 2):
            s = s + bs[i] * (x ** (-(1.0 * i + 2) / dimension))
        return s

    p0 = [0.0] * (FLAGS.knn_extrap_order - 2) + [(classes - 1.0) / classes]
    lower = [-1.0] * (FLAGS.knn_extrap_order - 2) + [0.0]
    # lower = [0.0]*(FLAGS.knn_extrap_order - 2 + 1)
    upper = [1.0] * (FLAGS.knn_extrap_order - 2) + [(classes - 1.0) / classes]
    params = curve_fit(paramed_curve, x, y, p0=p0, bounds=(lower, upper))
    return params[0][-1]


def eval_from_matrices(train_features, test_features, train_labels, test_labels):
    num_samples = train_features.shape[0]
    dimension = train_features.shape[1]
    x = [(num_samples // FLAGS.knn_extrap_samples) * (i + 1) for i in range(9)] + [
        num_samples
    ]

    classes = np.unique(np.concatenate((train_labels, test_labels))).size

    randomize = np.arange(num_samples)
    np.random.shuffle(randomize)
    train_features = train_features[randomize, :]
    train_labels = train_labels[randomize]

    results = {}
    for x_val in x:
        res = eval_from_matrices_split(
            train_features[:x_val, :], test_features, train_labels[:x_val], test_labels
        )
        for k, v in res.items():
            if k not in results.keys():
                results[k] = []
            results[k].append(v[0])

    final_results = {}
    # Extrapolate
    for key, y in results.items():
        k = int(key.split("k=")[1])
        extrap = extrapolate(x, y, classes, dimension)
        lower_ex = _get_lowerbound(extrap, k, classes)
        final_results[key] = [extrap, lower_ex]

    return final_results
