from absl import app
from absl import flags
from absl import logging
import csv
import importlib
import numpy as np
import os.path as path
import random
from sklearn.model_selection import train_test_split
import time

from transformations.reader.matrix import test_argument_and_file, load_and_log
import transformations.label_noise as label_noise
import methods.knn as knn
import methods.knn_extrapolate as knn_extrapolate
import methods.ghp as ghp
import methods.kde as kde
import methods.onenn as onenn
import methods.lr_model as lr_model

FLAGS = flags.FLAGS

flags.DEFINE_string("path", ".", "Path to the matrices directory")
flags.DEFINE_string(
    "features_train",
    None,
    "Name of the train features numpy matrix exported file (npy)",
)
flags.DEFINE_string(
    "features_test", None, "Name of the test features numpy matrix exported file (npy)"
)
flags.DEFINE_string(
    "labels_train", None, "Name of the train labels numpy matrix exported file (npy)"
)
flags.DEFINE_string(
    "labels_test", None, "Name of the test labels numpy matrix exported file (npy)"
)

flags.DEFINE_list(
    "noise_levels",
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Run at different noise levels",
)
flags.DEFINE_integer("noise_runs", 5, "Number of runs for different noise levels")

flags.DEFINE_string(
    "output_file", None, "File to write the output in CSV format (including headers)"
)
flags.DEFINE_bool(
    "output_overwrite",
    True,
    "Writes (if True) or appends (if False) to the specified output file if any",
)

flags.DEFINE_enum(
    "method",
    None,
    [
        "knn",
        "knn_loo",
        "knn_extrapolate",
        "ghp",
        "kde_knn_loo",
        "kde",
        "onenn",
        "lr_model",
    ],
    "Method to estimate the bayes error (results in either 1 value or a lower and upper bound)",
)


def _get_csv_row(variant, run, samples, noise, results, time):
    return {
        "method": FLAGS.method,
        "variant": variant,
        "run": run,
        "samples": samples,
        "noise": noise,
        "results": results,
        "time": time,
    }


def _write_result(rows):
    writeheader = False
    if FLAGS.output_overwrite or not path.exists(FLAGS.output_file):
        writeheader = True
    with open(FLAGS.output_file, mode="w+" if FLAGS.output_overwrite else "a+") as f:
        fieldnames = ["method", "variant", "run", "samples", "noise", "results", "time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if writeheader:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def estimate_from_split_matrices(eval_fn):
    test_argument_and_file(FLAGS.path, "features_train")
    test_argument_and_file(FLAGS.path, "features_test")
    test_argument_and_file(FLAGS.path, "labels_train")
    test_argument_and_file(FLAGS.path, "labels_test")

    train_features, dim_train, samples_train = load_and_log(
        FLAGS.path, "features_train"
    )
    test_features, dim_test, samples_test = load_and_log(FLAGS.path, "features_test")
    if dim_test != dim_train:
        raise AttributeError("Train and test features do not have the same dimension!")
    train_labels, dim, samples_train_labels = load_and_log(FLAGS.path, "labels_train")
    if dim != 1:
        raise AttributeError("Train labels file does not point to a vector!")
    if samples_train_labels != samples_train:
        raise AttributeError(
            "Train features and labels files does not have the same amount of samples!"
        )
    test_labels, _, samples_test_labels = load_and_log(FLAGS.path, "labels_test")
    if dim != 1:
        raise AttributeError("Test labels file does not point to a vector!")
    if samples_test_labels != samples_test:
        raise AttributeError(
            "Test features and labels files does not have the same amount of samples!"
        )

    logging.log(
        logging.DEBUG, "Start full estimation with method '{}'".format(FLAGS.method)
    )
    start = time.time()
    result_full = eval_fn(train_features, test_features, train_labels, test_labels)
    end = time.time()
    logging.log(
        logging.DEBUG,
        "Method '{}' executed in {} seconds".format(FLAGS.method, end - start),
    )
    logging.log(logging.INFO, "Full train and test set: {}".format(result_full))

    if FLAGS.noise_levels and FLAGS.noise_runs > 0:
        result_rows = []
        for run in range(FLAGS.noise_runs):
            if FLAGS.output_file:
                rows = [
                    _get_csv_row(
                        k,
                        run,
                        samples_train,
                        0.0,
                        v,
                        (end - start) / float(len(result_full)),
                    )
                    for k, v in result_full.items()
                ]
                result_rows.extend(rows)

            logging.log(
                logging.DEBUG,
                "Start noisy run {} out of {}".format(run + 1, FLAGS.noise_runs),
            )
            run_start = time.time()
            for noise_level in [float(x) for x in FLAGS.noise_levels]:
                if noise_level > 1.0 or noise_level <= 0.0:
                    raise AttributeError(
                        "Noise level {} has to be bigger than 0 and not larger than 1!".format(
                            noise_level
                        )
                    )
                logging.log(
                    logging.DEBUG,
                    "Start noise level {} for run {} out of {}".format(
                        noise_level, run + 1, FLAGS.noise_runs
                    ),
                )
                noise_start = time.time()

                # flip labels test and train
                flipped_train_labels = label_noise.random_flip(
                    train_labels, samples_train, noise_level, copy=True
                )
                flipped_test_labels = label_noise.random_flip(
                    test_labels, samples_test, noise_level, copy=True
                )

                # run method
                logging.log(
                    logging.DEBUG,
                    "Start full estimation with method '{}', noise level {}, run {}/{}".format(
                        FLAGS.method, noise_level, run + 1, FLAGS.noise_runs
                    ),
                )
                start = time.time()
                result = eval_fn(
                    train_features,
                    test_features,
                    flipped_train_labels,
                    flipped_test_labels,
                )
                end = time.time()
                logging.log(
                    logging.DEBUG,
                    "Method '{}' executed in {} seconds".format(
                        FLAGS.method, end - start
                    ),
                )
                logging.log(
                    logging.INFO,
                    "Run {}/{} - noise level {}: {}".format(
                        run + 1, FLAGS.noise_runs, noise_level, result
                    ),
                )

                if FLAGS.output_file:
                    rows = [
                        _get_csv_row(
                            k,
                            run,
                            samples_train,
                            noise_level,
                            v,
                            (end - start) / float(len(result)),
                        )
                        for k, v in result.items()
                    ]
                    result_rows.extend(rows)

                noise_end = time.time()
                logging.log(
                    logging.DEBUG,
                    "Noise level {} for run {}/{} executed in {} seconds".format(
                        noise_level, run + 1, FLAGS.noise_runs, noise_end - noise_start
                    ),
                )
            run_end = time.time()
            logging.log(
                logging.DEBUG,
                "Run {}/{} executed in {} seconds".format(
                    run + 1, FLAGS.noise_runs, run_end - run_start
                ),
            )

        if FLAGS.output_file:
            _write_result(result_rows)

    elif FLAGS.output_file:
        rows = [
            _get_csv_row(
                k, 0, samples_train, 0.0, v, (end - start) / float(len(result_full))
            )
            for k, v in result_full.items()
        ]
        _write_result(rows)


def estimate_from_single_matrix(eval_fn):
    test_argument_and_file(FLAGS.path, "features_train")
    test_argument_and_file(FLAGS.path, "labels_train")

    train_features, dim_train, samples_train = load_and_log(
        FLAGS.path, "features_train"
    )
    train_labels, dim, samples_train_labels = load_and_log(FLAGS.path, "labels_train")
    if dim != 1:
        raise AttributeError("Train labels file does not point to a vector!")
    if samples_train_labels != samples_train:
        raise AttributeError(
            "Train features and labels files does not have the same amount of samples!"
        )

    logging.log(
        logging.DEBUG, "Start full estimation with method '{}'".format(FLAGS.method)
    )
    start = time.time()
    result_full = eval_fn(train_features, train_labels)
    end = time.time()
    logging.log(
        logging.DEBUG,
        "Method '{}' executed in {} seconds".format(FLAGS.method, end - start),
    )
    logging.log(logging.INFO, "Full train set: {}".format(result_full))

    if FLAGS.noise_levels and FLAGS.noise_runs > 0:
        result_rows = []
        for run in range(FLAGS.noise_runs):
            if FLAGS.output_file:
                rows = [
                    _get_csv_row(
                        k,
                        run,
                        samples_train,
                        0.0,
                        v,
                        (end - start) / float(len(result_full)),
                    )
                    for k, v in result_full.items()
                ]
                result_rows.extend(rows)

            logging.log(
                logging.DEBUG,
                "Start noisy run {} out of {}".format(run + 1, FLAGS.noise_runs),
            )
            run_start = time.time()
            for noise_level in [float(x) for x in FLAGS.noise_levels]:
                if noise_level > 1.0 or noise_level <= 0.0:
                    raise AttributeError(
                        "Noise level {} has to be bigger than 0 and not larger than 1!".format(
                            noise_level
                        )
                    )
                logging.log(
                    logging.DEBUG,
                    "Start noise level {} for run {} out of {}".format(
                        noise_level, run + 1, FLAGS.noise_runs
                    ),
                )
                noise_start = time.time()

                # flip labels train
                flipped_train_labels = label_noise.random_flip(
                    train_labels, samples_train, noise_level, copy=True
                )

                # run method
                logging.log(
                    logging.DEBUG,
                    "Start full estimation with method '{}', noise level {}, run {}/{}".format(
                        FLAGS.method, noise_level, run + 1, FLAGS.noise_runs
                    ),
                )
                start = time.time()
                result = eval_fn(train_features, flipped_train_labels)
                end = time.time()
                logging.log(
                    logging.DEBUG,
                    "Method '{}' executed in {} seconds".format(
                        FLAGS.method, end - start
                    ),
                )
                logging.log(
                    logging.INFO,
                    "Run {}/{} - noise level {}: {}".format(
                        run + 1, FLAGS.noise_runs, noise_level, result
                    ),
                )

                if FLAGS.output_file:
                    rows = [
                        _get_csv_row(
                            k,
                            run,
                            samples_train,
                            noise_level,
                            v,
                            (end - start) / float(len(result)),
                        )
                        for k, v in result.items()
                    ]
                    result_rows.extend(rows)

                noise_end = time.time()
                logging.log(
                    logging.DEBUG,
                    "Noise level {} for run {}/{} executed in {} seconds".format(
                        noise_level, run + 1, FLAGS.noise_runs, noise_end - noise_start
                    ),
                )
            run_end = time.time()
            logging.log(
                logging.DEBUG,
                "Run {}/{} executed in {} seconds".format(
                    run + 1, FLAGS.noise_runs, run_end - run_start
                ),
            )

        if FLAGS.output_file:
            _write_result(result_rows)

    elif FLAGS.output_file:
        rows = [
            _get_csv_row(
                k, 0, samples_train, 0.0, v, (end - start) / float(len(result_full))
            )
            for k, v in result_full.items()
        ]
        _write_result(rows)


def main(argv):
    if FLAGS.method is None:
        raise app.UsageError("You have to specify the method!")

    if FLAGS.method == "knn":
        estimate_from_split_matrices(knn.eval_from_matrices)
    elif FLAGS.method == "knn_extrapolate":
        estimate_from_split_matrices(knn_extrapolate.eval_from_matrices)
    elif FLAGS.method == "lr_model":
        estimate_from_split_matrices(lr_model.eval_from_matrices)
    elif FLAGS.method == "knn_loo":
        estimate_from_single_matrix(knn.eval_from_matrix_loo)
    elif FLAGS.method == "ghp":
        estimate_from_single_matrix(ghp.eval_from_matrix)
    elif FLAGS.method == "kde_knn_loo":
        estimate_from_single_matrix(kde.eval_from_matrix_knn_loo)
    elif FLAGS.method == "onenn":
        estimate_from_single_matrix(onenn.eval_from_matrix_onenn)
    elif FLAGS.method == "kde":
        estimate_from_single_matrix(kde.eval_from_matrix_kde)
    else:
        raise NotImplementedError("Method module for 'matrices' not yet implemented!")


if __name__ == "__main__":
    app.run(main)
