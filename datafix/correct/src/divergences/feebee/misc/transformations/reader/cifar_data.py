from absl import app
from absl import flags
from absl import logging
import pickle
import numpy as np
import os.path as path
import struct
import time

FLAGS = flags.FLAGS

flags.DEFINE_string("cifar_path", ".", "Path to the features and labels cifar files")
flags.DEFINE_enum("cifar_type", None, ["train", "test"], "Cifar test or train files")
flags.DEFINE_enum("cifar_variant", "10", ["10", "100"], "Variant of CIFAR")


def test_file(filename):
    if not path.exists(path.join(FLAGS.cifar_path, filename)):
        raise app.UsageError(
            "File '{}' does not exists in the specified folder '{}'.".format(
                filename, FLAGS.cifar_path
            )
        )


def read():
    if not FLAGS.cifar_type:
        raise app.UsageError("--cifar_type needs to be defined!")

    train_batch_ids = [1, 2, 3, 4, 5]
    dim = 3072

    if FLAGS.cifar_variant == "10":
        if FLAGS.cifar_type == "train":
            filenames = ["data_batch_%d" % batchid for batchid in train_batch_ids]
            for fname in filenames:
                test_file(fname)

            samples = 50000

            data_f = {}
            data_l = {}

            for i, fname in enumerate(filenames):
                with open(path.join(FLAGS.cifar_path, fname), "rb") as fo:
                    u = pickle._Unpickler(fo)
                    u.encoding = "latin1"
                    d = u.load()
                    data_f[i] = d["data"]
                    data_l[i] = d["labels"]

            features = np.zeros((samples, dim))
            labels = np.zeros(
                samples,
            )

            for idx in range(len(filenames)):
                features[10000 * idx : 10000 * (idx + 1), :] = data_f[idx]
                labels[10000 * idx : 10000 * (idx + 1)] = data_l[idx]

            features = (
                features.reshape(-1, 3, 32, 32)
                .transpose((0, 2, 3, 1))
                .reshape(samples, -1)
            )

        else:
            filename = "test_batch"
            test_file(filename)

            samples = 10000

            with open(path.join(FLAGS.cifar_path, filename), "rb") as fo:
                u = pickle._Unpickler(fo)
                u.encoding = "latin1"
                d = u.load()
                features = d["data"]
                labels = np.array(d["labels"])

            features = (
                features.reshape(-1, 3, 32, 32)
                .transpose((0, 2, 3, 1))
                .reshape(samples, -1)
            )
    else:
        if FLAGS.cifar_type == "train":
            filename = "train"
            test_file(filename)

            samples = 50000

            with open(path.join(FLAGS.cifar_path, filename), "rb") as fo:
                u = pickle._Unpickler(fo)
                u.encoding = "latin1"
                d = u.load()
                features = d["data"]
                labels = np.array(d["fine_labels"])

            features = (
                features.reshape(-1, 3, 32, 32)
                .transpose((0, 2, 3, 1))
                .reshape(samples, -1)
            )
        else:
            filename = "test"
            test_file(filename)

            samples = 10000

            with open(path.join(FLAGS.cifar_path, filename), "rb") as fo:
                u = pickle._Unpickler(fo)
                u.encoding = "latin1"
                d = u.load()
                features = d["data"]
                labels = np.array(d["fine_labels"])

            features = (
                features.reshape(-1, 3, 32, 32)
                .transpose((0, 2, 3, 1))
                .reshape(samples, -1)
            )

    return features / 255.0, dim, samples, labels
