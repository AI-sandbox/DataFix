from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path
import struct
import time

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mnist_files_path", ".", "Path to the features and labels mnist files"
)
flags.DEFINE_string("features_mnist_file", None, "Name of the feature mnist file")
flags.DEFINE_string("labels_mnist_file", None, "Name of the labels mnist file")


def test_argument_and_file(folder_path, arg_name):
    if arg_name not in FLAGS.__flags.keys() or not FLAGS.__flags[arg_name].value:
        raise app.UsageError(
            "--{} is a required argument when runing the tool with the mnist file format.".format(
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


def read():
    test_argument_and_file(FLAGS.mnist_files_path, "features_mnist_file")
    test_argument_and_file(FLAGS.mnist_files_path, "labels_mnist_file")

    samples = path.join(FLAGS.mnist_files_path, FLAGS.features_mnist_file)
    labels = path.join(FLAGS.mnist_files_path, FLAGS.labels_mnist_file)

    with open(samples, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        features = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        features = features.reshape((size, nrows * ncols))
        dim = features.shape[1]
        samples = features.shape[0]

    with open(labels, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        if len(labels.shape) < 2:
            dim_labels = 1
        else:
            dim_labels = labels.shape[1]
        samples_labels = labels.shape[0]

    if dim_labels != 1:
        raise AttributeError("Labels file does not point to a vector!")
    if samples_labels != samples:
        raise AttributeError(
            "Features and labels files does not have the same amount of samples!"
        )

    return features / 255.0, dim, samples, labels
