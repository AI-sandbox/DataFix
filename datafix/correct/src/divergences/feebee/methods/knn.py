from absl import app
from absl import flags
from absl import logging
import numpy as np
import math
import time

from .utils import *

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "knn_measure",
    "squared_l2",
    ["squared_l2", "cosine"],
    "Measure used for KNN distance matrix computation",
)
flags.DEFINE_multi_integer("knn_k", "1", "Values for K in KNN", lower_bound=1)
flags.DEFINE_integer(
    "knn_subtest", None, "Split the testset for estimation", lower_bound=1
)
flags.DEFINE_integer(
    "knn_subtrain", None, "Split the trainset for estimation", lower_bound=1
)

KEY_PATTERN = "measure={0}, k={1}"


def _get_lowerbound(value, k, classes):
    if value <= 1e-10:
        return 0.0

    if classes > 2 or k == 1:
        return ((classes - 1.0) / float(classes)) * (
            1.0 - math.sqrt(max(0.0, 1 - ((float(classes) / (classes - 1.0)) * value)))
        )

    if k > 2:
        return value / float(1 + (1.0 / math.sqrt(k)))

    return value / float(1 + math.sqrt(2.0 / k))


def eval_from_folder(train_paths, test_paths, path_to_matrix_fn):
    if FLAGS.knn_subtrain is None:
        raise app.UsageError(
            "If train data is not split, there is no point in using variant 'folder' and not 'matrices'. Specify --knn_subtrain!"
        )

    total_classes = np.unique(
        np.concatenate((list(train_paths.keys()), list(test_paths.keys())))
    ).size
    vals = {}
    ks = sorted(set(FLAGS.knn_k), reverse=True)

    train_file_paths = [item for k, v in train_paths.items() for item in v]
    train_labels = np.array([k for k, v in train_paths.items() for item in v])

    test_file_paths = [item for k, v in test_paths.items() for item in v]
    test_labels = np.array([k for k, v in test_paths.items() for item in v])

    train_samples = len(train_file_paths)
    test_samples = len(test_file_paths)

    if FLAGS.knn_subtest is not None:
        test_dividable = (test_samples % FLAGS.knn_subtest) == 0
        test_iterations = (test_samples + FLAGS.knn_subtest - 1) // FLAGS.knn_subtest
    else:
        test_dividable = True
        test_iterations = 1

    for i in range(test_iterations):
        if test_iterations == 1:
            current_pos = 0
            current_samples = test_samples
        else:
            current_pos = i * FLAGS.knn_subtest
            current_samples = (
                min(test_samples, (i + 1) * FLAGS.knn_subtest) - current_pos
            )

        logging.log(
            logging.DEBUG,
            "Start estimation test iteration {}/{}".format(i + 1, test_iterations),
        )

        logging.log(
            logging.DEBUG,
            "{}/{} - Start loading feature test matrix".format(i + 1, test_iterations),
        )

        test_features = path_to_matrix_fn(
            test_file_paths[current_pos : (current_pos + current_samples)]
        )

        # select per split to nearest k (for max k) train point distances and indices
        train_iterations = (
            train_samples + FLAGS.knn_subtrain - 1
        ) // FLAGS.knn_subtrain
        d = None
        y_train = None
        for j in range(train_iterations):
            start_idx = j * FLAGS.knn_subtrain
            end_idx = min(train_samples, (j + 1) * FLAGS.knn_subtrain)
            logging.log(
                logging.DEBUG,
                "{}/{} - Start loading feature train matrix {}/{}".format(
                    i + 1, test_iterations, j + 1, train_iterations
                ),
            )
            train_features = path_to_matrix_fn(train_file_paths[start_idx:end_idx])
            logging.log(
                logging.DEBUG,
                "{}/{} - Start computing distance sub matrix {}/{} with measure='{}'".format(
                    i + 1, test_iterations, j + 1, train_iterations, FLAGS.knn_measure
                ),
            )
            start = time.time()
            d_j = compute_distance_matrix(
                train_features, test_features, FLAGS.knn_measure
            )
            end = time.time()
            logging.log(
                logging.DEBUG,
                "{}/{} - Sub D computed in {} seconds".format(
                    i + 1, test_iterations, end - start
                ),
            )

            if end_idx - start_idx < ks[0]:
                # do not run argpartition
                sub_d = d_j
                indices = np.tile(range(start_idx, end_idx), (sub_d.shape[0], 1))
            else:
                # run argpartition and build
                indices = np.argpartition(d_j, ks[0] - 1, axis=1)
                # update sub_d and y_train_new
                num_rows = indices[:, : ks[0]].shape[0]
                num_cols = indices[:, : ks[0]].shape[1]
                rows = [x for x in range(num_rows) for _ in range(num_cols)]
                cols = indices[:, : ks[0]].reshape(-1)
                sub_d = d_j[rows, cols].reshape(num_rows, -1)

                indices = indices[:, : ks[0]] + start_idx

            y_train_new = indices
            for k in range(y_train_new.shape[0]):
                y_train_new[k, :] = train_labels[y_train_new[k, :]]

            if d is None:
                d = sub_d
            else:
                d = np.concatenate((d, sub_d), axis=1)

            if y_train is None:
                y_train = y_train_new
            else:
                y_train = np.concatenate((y_train, y_train_new), axis=1)

        # run knn on that smaller matrix
        logging.log(
            logging.DEBUG,
            "{}/{} - Start computing KNN error for ks={}".format(
                i + 1, test_iterations, ks
            ),
        )
        start = time.time()
        err = knn_errorrate(
            d, y_train, test_labels[current_pos : (current_pos + current_samples)], k=ks
        )
        end = time.time()
        logging.log(
            logging.DEBUG,
            "{}/{} - KNN error computed in {} seconds".format(
                i + 1, test_iterations, end - start
            ),
        )

        if not test_dividable:
            err = [e * (current_samples / test_features.shape[0]) for e in err]
        else:
            err = [e * (1.0 / test_iterations) for e in err]

        for idx, k in enumerate(ks):
            if k not in vals:
                vals[k] = err[idx]
            else:
                vals[k] += err[idx]

    results = {}
    for k, v in vals.items():
        results[KEY_PATTERN.format(FLAGS.knn_measure, k)] = [
            v,
            _get_lowerbound(v, k, total_classes),
        ]

    return results


def eval_from_matrices_split(train_features, test_features, train_labels, test_labels):
    total_classes = np.unique(np.concatenate((train_labels, test_labels))).size
    vals = {}
    ks = sorted(set(FLAGS.knn_k), reverse=True)

    if FLAGS.knn_subtest is not None:
        test_dividable = (test_features.shape[0] % FLAGS.knn_subtest) == 0
        test_iterations = (
            test_features.shape[0] + FLAGS.knn_subtest - 1
        ) // FLAGS.knn_subtest
    else:
        test_dividable = True
        test_iterations = 1

    for i in range(test_iterations):
        if test_iterations == 1:
            current_pos = 0
            current_samples = test_features.shape[0]
        else:
            current_pos = i * FLAGS.knn_subtest
            current_samples = (
                min(test_features.shape[0], (i + 1) * FLAGS.knn_subtest) - current_pos
            )

        logging.log(
            logging.DEBUG,
            "Start estimation test iteration {}/{}".format(i + 1, test_iterations),
        )

        # if multiple train iterations
        if FLAGS.knn_subtrain is not None:
            # select per split to nearest k (for max k) train point distances and indices
            train_iterations = (
                train_features.shape[0] + FLAGS.knn_subtrain - 1
            ) // FLAGS.knn_subtrain
            d = None
            y_train = None
            for j in range(train_iterations):
                start_idx = j * FLAGS.knn_subtrain
                end_idx = min(train_features.shape[0], (j + 1) * FLAGS.knn_subtrain)
                logging.log(
                    logging.DEBUG,
                    "{}/{} - Start computing distance sub matrix {}/{} with measure='{}'".format(
                        i + 1,
                        test_iterations,
                        j + 1,
                        train_iterations,
                        FLAGS.knn_measure,
                    ),
                )
                start = time.time()
                d_j = compute_distance_matrix(
                    train_features[start_idx:end_idx, :],
                    test_features[current_pos : (current_pos + current_samples), :],
                    FLAGS.knn_measure,
                )
                end = time.time()
                logging.log(
                    logging.DEBUG,
                    "{}/{} - Sub D computed in {} seconds".format(
                        i + 1, test_iterations, end - start
                    ),
                )

                if end_idx - start_idx < ks[0]:
                    # do not run argpartition
                    sub_d = d_j
                    indices = np.tile(range(start_idx, end_idx), (sub_d.shape[0], 1))
                else:
                    # run argpartition and build
                    indices = np.argpartition(d_j, ks[0] - 1, axis=1)
                    # update sub_d and y_train_new
                    num_rows = indices[:, : ks[0]].shape[0]
                    num_cols = indices[:, : ks[0]].shape[1]
                    rows = [x for x in range(num_rows) for _ in range(num_cols)]
                    cols = indices[:, : ks[0]].reshape(-1)
                    sub_d = d_j[rows, cols].reshape(num_rows, -1)

                    indices = indices[:, : ks[0]] + start_idx

                y_train_new = indices
                for k in range(y_train_new.shape[0]):
                    y_train_new[k, :] = train_labels[y_train_new[k, :]]

                if d is None:
                    d = sub_d
                else:
                    d = np.concatenate((d, sub_d), axis=1)

                if y_train is None:
                    y_train = y_train_new
                else:
                    y_train = np.concatenate((y_train, y_train_new), axis=1)

            # run knn on that smaller matrix
            logging.log(
                logging.DEBUG,
                "{}/{} - Start computing KNN error for ks={}".format(
                    i + 1, test_iterations, ks
                ),
            )
            start = time.time()
            err = knn_errorrate(
                d,
                y_train,
                test_labels[current_pos : (current_pos + current_samples)],
                k=ks,
            )
            end = time.time()
            logging.log(
                logging.DEBUG,
                "{}/{} - KNN error computed in {} seconds".format(
                    i + 1, test_iterations, end - start
                ),
            )

        else:
            logging.log(
                logging.DEBUG,
                "{}/{} - Start computing distance matrix with measure='{}'".format(
                    i + 1, test_iterations, FLAGS.knn_measure
                ),
            )
            start = time.time()
            d = compute_distance_matrix(
                train_features,
                test_features[current_pos : (current_pos + current_samples), :],
                FLAGS.knn_measure,
            )
            end = time.time()
            logging.log(
                logging.DEBUG,
                "{}/{} - D computed in {} seconds".format(
                    i + 1, test_iterations, end - start
                ),
            )

            logging.log(
                logging.DEBUG,
                "{}/{} - Start computing KNN error for ks={}".format(
                    i + 1, test_iterations, ks
                ),
            )
            start = time.time()
            err = knn_errorrate(
                d,
                train_labels,
                test_labels[current_pos : (current_pos + current_samples)],
                k=ks,
            )
            end = time.time()
            logging.log(
                logging.DEBUG,
                "{}/{} - KNN error computed in {} seconds".format(
                    i + 1, test_iterations, end - start
                ),
            )

        if not test_dividable:
            err = [e * (current_samples / test_features.shape[0]) for e in err]
        else:
            err = [e * (1.0 / test_iterations) for e in err]

        for idx, k in enumerate(ks):
            if k not in vals:
                vals[k] = err[idx]
            else:
                vals[k] += err[idx]

    results = {}
    for k, v in vals.items():
        results[KEY_PATTERN.format(FLAGS.knn_measure, k)] = [
            v,
            _get_lowerbound(v, k, total_classes),
        ]

    return results


def eval_from_matrices(train_features, test_features, train_labels, test_labels):
    knn_measure = "squared_l2"
    knn_k = [1, 2, 3, 4, 5, 10, 20, 50]
    # if FLAGS.knn_subtest is not None or FLAGS.knn_subtrain is not None:
    #     return eval_from_matrices_split(train_features, test_features, train_labels, test_labels)

    # logging.log(logging.DEBUG, "Start computing distance matrix with measure='{}'".format(FLAGS.knn_measure))
    start = time.time()
    d = compute_distance_matrix(train_features, test_features, knn_measure)
    end = time.time()
    logging.log(logging.DEBUG, "D computed in {} seconds".format(end - start))

    total_classes = np.unique(np.concatenate((train_labels, test_labels))).size

    results = {}

    ks = sorted(set(knn_k), reverse=True)
    logging.log(logging.DEBUG, "Start computing KNN error for ks={}".format(ks))
    start = time.time()
    err = knn_errorrate(d, train_labels, test_labels, k=ks)
    end = time.time()
    logging.log(logging.DEBUG, "KNN error computed in {} seconds".format(end - start))
    for idx, k in enumerate(ks):
        results[f"knn_{k}"] = [err[idx], _get_lowerbound(err[idx], k, total_classes)]

    return results


def eval_from_matrix_loo(train_features, train_labels):
    logging.log(
        logging.DEBUG,
        "Start computing LOO distance matrix with measure='{}'".format(
            FLAGS.knn_measure
        ),
    )
    start = time.time()
    d = compute_distance_matrix_loo(train_features, FLAGS.knn_measure)
    end = time.time()
    logging.log(logging.DEBUG, "D computed in {} seconds".format(end - start))

    total_classes = np.unique(train_labels).size

    results = {}

    ks = sorted(set(FLAGS.knn_k), reverse=True)
    logging.log(logging.DEBUG, "Start computing LOO KNN error for ks={}".format(ks))
    start = time.time()
    err = knn_errorrate_loo(d, train_labels, k=ks)
    end = time.time()
    logging.log(
        logging.DEBUG, "LOO KNN error computed in {} seconds".format(end - start)
    )
    for idx, k in enumerate(ks):
        results[KEY_PATTERN.format(FLAGS.knn_measure, k)] = [
            err[idx],
            _get_lowerbound(err[idx], k, total_classes),
        ]

    return results
