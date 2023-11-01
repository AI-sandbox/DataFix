from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path
import time

from . import generic

FLAGS = flags.FLAGS

flags.DEFINE_string("matrices_path", ".", "Path to the features and labels matrices")
flags.DEFINE_string("features_matrix", None, "Name of the feature matrix to read")
flags.DEFINE_string(
    "labels_matrix", None, "Name of the labels matrix (or vector) to read"
)
flags.DEFINE_integer("input_height", 299, "Input height of the matrix")
flags.DEFINE_integer("input_width", 299, "Input width of the matrix")
flags.DEFINE_integer("input_channels", 3, "Input channels of the matrix")


def test_argument_and_file(folder_path, arg_name):
    if arg_name not in FLAGS.__flags.keys() or not FLAGS.__flags[arg_name].value:
        raise app.UsageError(
            "--{} is a required argument when runing the tool with matrices.".format(
                arg_name
            )
        )

    arg_val = FLAGS.__flags[arg_name].value

    if not path.exists(path.join(folder_path, arg_val)):
        raise app.UsageError(
            "File '{}' given by '--{}' does not exists in the specified folder '{}'.".format(
                arg_val, arg_name, folder_path
            )
        )


def load_and_log(folder_path, arg_name):
    arg_val = FLAGS.__flags[arg_name].value

    logging.log(logging.DEBUG, "Start loading numpy array '{}'".format(arg_name))
    start = time.time()
    matrix = np.load(path.join(folder_path, arg_val))
    end = time.time()
    logging.log(
        logging.DEBUG, "'{}' loaded in {} seconds".format(arg_name, end - start)
    )

    if len(matrix.shape) > 2:
        raise AttributeError(
            "Argument '--{}' points to a tensor and not a matrix!".format(arg_name)
        )

    if len(matrix.shape) == 1:
        logging.log(
            logging.INFO,
            "'{}' loaded as a vector with '{}' samples".format(
                arg_val, matrix.shape[0]
            ),
        )
        return matrix, 1, matrix.shape[0]

    logging.log(
        logging.INFO,
        "'{}' loaded as matrix with dimension '{}' and '{}' samples".format(
            arg_val, matrix.shape[1], matrix.shape[0]
        ),
    )
    return matrix, matrix.shape[1], matrix.shape[0]


def read():
    test_argument_and_file(FLAGS.matrices_path, "features_matrix")
    test_argument_and_file(FLAGS.matrices_path, "labels_matrix")

    features, dim, samples = load_and_log(FLAGS.matrices_path, "features_matrix")
    labels, dim_labels, samples_labels = load_and_log(
        FLAGS.matrices_path, "labels_matrix"
    )
    if dim_labels != 1:
        raise AttributeError("Labels file does not point to a vector!")
    if samples_labels != samples:
        raise AttributeError(
            "Features and labels files does not have the same amount of samples!"
        )

    return features, dim, samples, labels


def apply_fn_matrices(features, dim, samples, labels, fn, use_output=True):
    indices = list(range(samples))
    num_batches = int(samples + (FLAGS.batch_size - 1)) // FLAGS.batch_size
    new_features = []

    cnt = 0
    for select in generic.chunks(indices, FLAGS.batch_size, samples):
        if cnt == 0 or cnt % 10 == 9 or cnt == num_batches - 1:
            logging.log(
                logging.INFO, "Matrix -- Processing {}/{}".format(cnt + 1, num_batches)
            )

        images = features[select, :].reshape(
            len(select), FLAGS.input_height, FLAGS.input_width, FLAGS.input_channels
        )
        if FLAGS.input_channels == 1:
            images = np.repeat(images, 3, axis=3)

        if use_output:
            res = fn(images)

            new_features.extend(res.reshape(res.shape[0], -1))

            # ONLY NEEDED IF SHUFFLING
            # y.extend(labels[select])
        else:
            fn(images)

        cnt += 1

    if not use_output:
        return

    features = np.array(new_features)
    samples = features.shape[0]
    dim = features.shape[1]

    if len(labels.shape) != 1:
        raise AttributeError("Labels file does not point to a vector!")
    if labels.shape[0] != samples:
        raise AttributeError(
            "Features and labels files does not have the same amount of samples!"
        )

    return features, dim, samples, labels
